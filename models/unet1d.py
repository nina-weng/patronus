
import math
import torch
import torch.nn as nn

def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        """
        Args:
          total_time_steps: maximum diffusion steps (T)
          time_emb_dims: dimension of the raw sinusoidal embedding
          time_emb_dims_exp: expanded dimension after MLP
        """
        super().__init__()

        half_dim = time_emb_dims // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_scale)

        # Precompute embeddings for all time steps
        ts = torch.arange(total_time_steps, dtype=torch.float32)
        emb = ts.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # shape: (total_time_steps, time_emb_dims)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=True),        # [T] -> [time_emb_dims]
            nn.Linear(time_emb_dims, time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(time_emb_dims_exp, time_emb_dims_exp),
        )

    def forward(self, time):
        """
        Args:
          time: shape [batch], containing the integer timesteps
        Returns:
          Tensor of shape [batch, time_emb_dims_exp]
        """
        return self.time_blocks(time)


class ResBlock1D(nn.Module):
    """
    A simple 1D residual block that incorporates time embeddings.
    - Each block has:
        (GroupNorm -> SiLU -> Conv1D) -> + time embedding
        (GroupNorm -> SiLU -> Conv1D)
        + skip connection
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=512, dropout_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # time embedding linear
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # if in/out channels differ, learn a 1x1 conv to match shapes
        self.skip_connection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) 
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        """
        x: (batch, in_channels, length)
        t_emb: (batch, time_emb_dim)
        """
        # First half
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Inject time embedding (broadcast over length dimension)
        # shape of t_emb: [B, time_emb_dim], we project to out_channels, then unsqueeze
        time_embedding = self.time_emb_proj(t_emb)  # [B, out_channels]
        time_embedding = time_embedding.unsqueeze(-1)  # [B, out_channels, 1]
        h = h + time_embedding

        # Second half
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip_connection(x) + h


class DownSample1D(nn.Module):
    """
    Downsample by a factor of 2 using stride=2 conv or pooling.
    """
    def __init__(self, channels):
        super().__init__()
        
        self.down = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)
    



class UpSample1D(nn.Module):
    """
    Upsample by a factor of 2. We can use transposed conv or interpolation + conv.
    Below uses a simple transposed conv approach.
    """
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1,)
        # self.up = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1,output_padding=1)

    def forward(self, x):
        return self.up(x)


class UNet1D(nn.Module):
    """
    Minimal UNet in 1D for vector inputs:
      - x: shape (B, input_channels=1, length=100) as an example
      - time t: shape (B,) of timesteps -> fed into time embedding
    """
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        base_channels=64,
        channel_mults=[1, 2],
        num_res_blocks=2,
        dropout_rate=0.1,
        total_time_steps=1000,
    ):
        """
        Args:
          input_channels:  number of channels in input (for pure vectors, typically 1)
          output_channels: number of channels in output
          base_channels:   number of base channels in first layer
          channel_mults:   how we scale the channel count at each downsampling
          num_res_blocks:  how many ResBlocks per resolution level
          dropout_rate:    dropout in the ResBlock
          total_time_steps: maximum diffusion steps for time embedding
        """
        super().__init__()

        # 1) Time Embeddings
        # We'll embed times into a (base_channels*4) dimension, for instance
        self.time_emb = SinusoidalPositionEmbeddings(
            total_time_steps=total_time_steps,
            time_emb_dims=base_channels, 
            time_emb_dims_exp=base_channels * 4
        )

        # 2) Initial projection
        self.initial_conv = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)

        # 3) Encoder (downsampling) layers
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        self.resolutions = []
        
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            # Add num_res_blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock1D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    time_emb_dim=base_channels * 4,
                    dropout_rate=dropout_rate,
                ))
                current_channels = out_channels
            # add DownSample except on last
            if i < len(channel_mults) - 1:
                self.down_blocks.append(DownSample1D(current_channels))
        
        # 4) Bottleneck (one or two ResBlocks at final resolution)
        self.mid_block1 = ResBlock1D(
            in_channels=current_channels,
            out_channels=current_channels,
            time_emb_dim=base_channels * 4,
            dropout_rate=dropout_rate
        )
        self.mid_block2 = ResBlock1D(
            in_channels=current_channels,
            out_channels=current_channels,
            time_emb_dim=base_channels * 4,
            dropout_rate=dropout_rate
        )

        # 5) Decoder (upsampling) layers
        self.up_blocks = nn.ModuleList()
        channel_mults_rev = list(reversed(channel_mults))
        for i, mult in enumerate(channel_mults_rev):
            out_channels = base_channels * mult
            # Add num_res_blocks + possible skip connections
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResBlock1D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    time_emb_dim=base_channels * 4,
                    dropout_rate=dropout_rate
                ))
                current_channels = out_channels
            # add UpSample except on last
            if i < len(channel_mults_rev) - 1:
                self.up_blocks.append(UpSample1D(current_channels))

        # 6) Final projection
        self.final_conv = nn.Conv1d(current_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: (B, input_channels, length)
        t: (B,) integer timesteps
        """
        # Embed time
        t_emb = self.time_emb(t)  # shape: [B, base_channels*4]

        # Initial
        x = self.initial_conv(x)

        # Down path
        skips = []
        h = x
        for layer in self.down_blocks:
            if isinstance(layer, DownSample1D):
                # store skip before downsampling
                skips.append(h)
                h = layer(h)
                # print(f'down: {h.shape=}')
            else:
                h = layer(h, t_emb=t_emb)
        # final skip
        skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb=t_emb)
        # print(f'Bottleneck: {h.shape=}')
        h = self.mid_block2(h, t_emb=t_emb)
        # print(f'Bottleneck: {h.shape=}')

        

        # Up path
        for layer in self.up_blocks:
            if isinstance(layer, UpSample1D):
                # pop skip
                h_skip = skips.pop()
                # optional: combine skip features, e.g. cat() 
                # but for a minimal example, we won't cat (or do residual merges).
                # If you want, do: h = torch.cat([h, h_skip], dim=1) 
                h = layer(h)
            else:
                h = layer(h, t_emb=t_emb)

        # Final
        h = self.final_conv(h)
                
        return h
    




if __name__ == "__main__":
    B = 8
    L = 100
    x = torch.randn(B, 1, L)      # (batch=8, channels=1, length=100)
    t = torch.randint(0, 200, (B,))  # timesteps in [0, 200)

    model = UNet1D(
        input_channels=1,
        output_channels=1,
        base_channels=32,      # smaller for demonstration
        channel_mults=[1, 2],  # two resolution levels
        num_res_blocks=1,      # fewer blocks for simplicity
        dropout_rate=0.1,
        total_time_steps=200,  # for the sinusoidal embedding
    )

    out = model(x, t)
    print("Input shape:", x.shape)  # (8, 1, 100)
    print("Output shape:", out.shape)  # (8, 1, 100)
    print(model)
