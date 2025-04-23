

import torch
import torch.nn as nn
import math

from models.encoders_basic import ProtoActLearningBlock

def zero_module(module: nn.Module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        ts = torch.arange(total_time_steps, dtype=torch.float32)
        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)

class VectorConditioning(nn.Module):
    def __init__(self, num_proto, cond_emb_dims_exp):
        super().__init__()
        self.vector_fc = nn.Sequential(
            nn.Linear(num_proto, cond_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(cond_emb_dims_exp, cond_emb_dims_exp)
        )

    def forward(self, cond_vector):
        return self.vector_fc(cond_vector)
    

class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
        h = h.swapaxes(2, 1).view(B, self.channels, H, W)  # [B, C, H*W] --> [B, C, H, W]
        return x + h



class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512, apply_attention=False, attention_type='ori_att'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_fn = nn.SiLU()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels) # for the time information
        
        self.dense_2 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels) # for the pact information
        
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            if attention_type == 'ori_att':
                self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t, cond_emb):
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)

        h += self.dense_1(self.act_fn(t))[:, :, None, None]
        h += self.dense_2(self.act_fn(cond_emb))[:, :, None, None]


        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)
        h = h + self.match_input(x)
        h = self.attention(h)
        return h


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)



class Patronus_Unet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 output_channels=3, 
                 num_res_blocks=2, 
                 base_channels=128, 
                 base_channels_multiples=(1, 2, 4, 8), 
                 apply_attention=(False, False, True, False),
                 attention_type='ori_att', 
                 dropout_rate=0.1, 
                 time_multiple=4, 
                # patronus
                 num_proto=15,
                 prototype_vector_shape=(1,128),
                 encoder_type='basic_conv',
                 img_size = (3, 64, 64),
                 plb_channel_inputs = [1, 32, 64, 64, 128],
                 plb_layer_filter_sizes = [3,3,3,3],
                 plb_layer_strides = [2,1,1,1],
                 plb_layer_paddings = [1,0,0,0],
                 ):
        super().__init__()

        self.attention_type = attention_type
        assert self.attention_type in ['ori_att', 'simple_att']
        self.num_proto = num_proto # number of prototypes
        self.prototype_vector_shape = prototype_vector_shape
        assert self.prototype_vector_shape[0] == 1, 'Currently only support: the first dimension of prototype vector should be 1' 
        self.input_channels = input_channels
        self.plb_channel_inputs = plb_channel_inputs
        self.plb_layer_filter_sizes = plb_layer_filter_sizes
        self.plb_layer_strides = plb_layer_strides
        self.plb_layer_paddings = plb_layer_paddings
        self.img_size = img_size

        self.encoder_type = encoder_type
        self.proactBlock = ProtoActLearningBlock(
                num_prototypes=self.num_proto , 
                encoder_type = self.encoder_type,
                input_channels =self.input_channels,
                img_size = self.img_size[2],
                channel_inputs = self.plb_channel_inputs, 
                layer_filter_sizes=self.plb_layer_filter_sizes,
                layer_strides = self.plb_layer_strides,
                layer_paddings = self.plb_layer_paddings,
        )
        assert self.encoder_type in ['basic_conv'], 'Currently only support: basic_conv'
        # TODO: generalize to other more basic frameworks like resnet, densenet, etc.

        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)
        self.cond_embeddings = VectorConditioning(num_proto=num_proto, cond_emb_dims_exp=time_emb_dims_exp)
        self.first = nn.Conv2d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")

        num_resolutions = len(base_channels_multiples)
        self.encoder_blocks = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]
            for _ in range(num_res_blocks):
                block = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                    attention_type=attention_type,
                )
                self.encoder_blocks.append(block)
                in_channels = out_channels
                curr_channels.append(in_channels)
            if level != (num_resolutions - 1):
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=True,
                    attention_type=attention_type,
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False,
                    attention_type=attention_type,
                ),
            )
        )

        self.decoder_blocks = nn.ModuleList()
        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]
            for _ in range(num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                    attention_type=attention_type,
                )
                in_channels = out_channels
                self.decoder_blocks.append(block)
            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels))

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=3, stride=1, padding="same"),
        )



    def forward(self, x, t, given_cond_vector=None):
        time_emb = self.time_embeddings(t)
        if given_cond_vector is None:
            # training pharse
            cond_vector = self.proactBlock(x)
        else:
            # testing pharse
            cond_vector = given_cond_vector
        cond_emb = self.cond_embeddings(cond_vector)

        h = self.first(x)
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h, time_emb, cond_emb)
            outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb, cond_emb)

        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb, cond_emb)

        h = self.final(h)
        return h
    

    def get_proact(self,x):
        '''
        get the prototype activation
        '''
        return self.proactBlock(x)


    def get_patch_size(self):
        proto_layer_rf_info = self.proactBlock.proto_layer_rf_info
        return proto_layer_rf_info

    def get_learned_prototypes(self):
        # return the learned prototypes, notice that it should be detached, otehrwise the gradident graph will be changed
        return self.proactBlock.prototype_vectors
    




# test
if __name__ == '__main__':
    print('Test the functionality of the model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device=}')

    # x = torch.randn(2, 1, 32, 32).to(device)
    # x = torch.randn(2, 3, 64, 64).to(device)
    x = torch.randn(4, 1, 224, 224).to(device) # chexpert size


    model = Patronus_Unet(input_channels = x.shape[1],
                            output_channels= x.shape[1], 
                            plb_channel_inputs = [1, 32, 64, 64, 128, 128],
                            plb_layer_filter_sizes = [5,5,5,5,5],
                            plb_layer_strides = [2,2,1,1,1],
                            plb_layer_paddings = [1,1,0,0,0],
                            
                         )
    model.to(device)
    # print the model
    print(model)


    
    # get the corret time tensor
    ts = torch.randint(low=1, high=1000, size=(x.shape[0],), device=device)
    print(f'{x.shape=},{ts.shape=},{x.device=},{ts.device=}')

    y = model(x, ts)
    print(f'{y.shape=}')
    # print(y)

    proact = model.get_proact(x,ts)
    print(f'{proact.shape=}')

    hidden = model.proactBlock.encoder(x,ts)
    print(f'{hidden.shape=}')

    print(f'{model.get_patch_size()=}')

    print('forward function is okay')