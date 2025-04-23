'''
training an additional DDIM for sampling latent space, which is p_act in our model
This DDIM needed to be trained after the training of semantic encoder and diffusion models for the images, 
So we need the config information from previous training
and we also need to load the pre-trained semantic encoder to get the p_act for trainging this DDIM

Note: 
1. we use a simple UNet here
2. using DDIM for training and inference as DiffAuto paper

run this by `python ./train/train_latent_ddim.py --dataset CelebA --version_num 4`
'''
import sys
sys.path.append('../patronus/')

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.cuda import amp
import numpy as np
import os
import torch.optim as optim
import wandb
from torchmetrics import MeanMetric
import argparse
import wandb
from torchmetrics import MeanMetric
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from tqdm import tqdm


from train.utils import load_patronus_unet_model
from train.dataloader import get_dataloader_pact
from models.unet1d import UNet1D
from global_config import * 


class DDIMSampler:
    def __init__(self, model, device, num_timesteps=1000, eta=0.0,beta_schedule='linear'):
        self.model = model
        self.num_timesteps = num_timesteps
        self.eta = eta
        self.device = device
        self.beta_schedule = beta_schedule
        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        if self.beta_schedule == 'linear':
            self.beta  = self.get_betas()
        elif self.beta_schedule == 'cosine':
            self.beta  = self.get_betas_cosine()
        self.alpha = 1 - self.beta
        
        self.sqrt_beta                       = torch.sqrt(self.beta)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)
    

    def sample_ddim(self, shape):
        with torch.no_grad():
            x = torch.randn(shape, device=self.device)  # Initialize latent with noise
            for i in reversed(range(self.num_timesteps)):
                t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
                
                # Predict noise using the model
                predicted_noise = self.model(x, t)
                
                # Use precomputed alpha and alpha_bar
                alpha = self.alpha[i]
                alpha_bar = self.alpha_cumulative[i]
                if i > 0:
                    alpha_bar_prev = self.alpha_cumulative[i - 1]
                    beta = self.beta[i]
                    noise = torch.randn_like(x)

                    # Compute x_t-1 using DDIM formula
                    x = (
                        torch.sqrt(alpha_bar_prev) * ((x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar))
                        + torch.sqrt(1 - alpha_bar_prev - self.eta**2 * beta) * noise
                    )
                else:
                    # Final step: deterministic prediction
                    x = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        return x
    
    def sample_ddpm(self, shape):
        with torch.no_grad():
            x = torch.randn(shape, device=self.device)  # Initialize latent with noise

            for i in reversed(range(self.num_timesteps)):
                t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)

                # Predict noise using the model
                predicted_noise = self.model(x, t)

                # Use precomputed alpha and alpha_bar
                alpha = self.alpha[i]
                alpha_bar = self.alpha_cumulative[i]
                if i > 0:
                    alpha_bar_prev = self.alpha_cumulative[i - 1]
                    beta = self.beta[i]
                    noise = torch.randn_like(x)

                    # Compute x_t-1 using DDPM formula
                    x = (
                        torch.sqrt(alpha_bar_prev) * ((x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar))
                        + torch.sqrt(beta) * noise
                    )
                else:
                    # Final step: deterministic prediction
                    x = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        return x


    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
    
    def get_betas_cosine(self):
        steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1, dtype=torch.float32)
        res = torch.cos((steps / self.num_timesteps) * (np.pi / 2)) ** 2
        res = res.to(self.device)
        return res
    
    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        eps = torch.randn_like(x0)  # Noise
        def get(element: torch.Tensor, t: torch.Tensor):
            """
            Get value at index position "t" in "element" and
                reshape it to have the same dimension as a batch of images.
            """
            ele = element.gather(-1, t)
            return ele.reshape(-1, 1,1)
        mean    = get(self.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
        std_dev = get(self.sqrt_one_minus_alpha_cumulative, t=timesteps) 

        sample  = mean + std_dev * eps # scaled inputs * scaled noise

        return sample, eps  # return ... , gt noise --> model predicts this)
    

# ------------------------
# Training Loop
# ------------------------
def train_ddim(model, sd, scaler, dataloader, batch_size, optimizer, num_timesteps, device, epochs, save_path):


    # Initialize wandb
    project_name = 'latent_ddim'
    wandb.init(project=project_name, config={
        "num_timesteps": num_timesteps,
        "epochs": epochs,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": batch_size,
    })

    model.train()
    mse_loss = nn.MSELoss()
    best_loss = float('inf')  # Initialize best loss to a very large value

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        loss_record = MeanMetric()

        for ind, batch in enumerate(loop):
            x0s, _ = batch
            x0s = x0s.to(device)
            ts = torch.randint(low=1, high=num_timesteps, size=(x0s.shape[0],), device=device)
            xts, gt_noise = sd.forward_diffusion(x0s, ts)

            # print(f'DEBUG:{x0s.shape=},{xts.shape=},{gt_noise.shape=}')

            with amp.autocast():
                pred_noise = model(xts, ts)
                # print(f'DEBUG:{pred_noise.shape=}')
                loss = mse_loss(gt_noise, pred_noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            wandb.log({"Loss-train-step": loss_value})

        # Compute the mean loss for the epoch
        mean_loss = loss_record.compute().item()
        wandb.log({"epoch": epoch + 1, "loss": mean_loss})

        # Save model checkpoint only if the loss improves
        if mean_loss < best_loss:
            best_loss = mean_loss
            checkpoint_path = f"{save_path}/best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path} with loss {best_loss:.6f}")

    # Save the final model after all epochs
    final_model_path = f"{save_path}/final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    wandb.finish()

def get_version_number(save_path):
    if os.path.isdir(save_path):
        # Get all folders numbers in the root_log_dir
        folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(save_path)]
        
        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)

        return last_version_number +1
    else: return 0



def main(ds, version_num,
         training_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # 1. get the pre-trained semantic encoder
    print('*'*30 + 'Load model' + '*'*30)
    model, patronus_config_set = load_patronus_unet_model(ds_name=ds, 
                                                        version_num=version_num,
                                )
    
    model.eval()
    model.to(device)

    prototype_encoder = model.proactBlock
    prototype_encoder.eval()
    prototype_encoder.to(device=device)
    prototype_encoder.eval()

    # 2. get the training data
    
    # 2.1 get the prototype activation for the training data and wrap it as dataloader
    dataloader_train_pact = get_dataloader_pact(dataset_name=f'{ds}-train',
        batch_size=64,
        pact_encoder=prototype_encoder,
        device=device,
        shuffle = False,  # do not need to shuffle here
    )



    # 3. Initialize model and optimizer
    # using the defult parameters for the UNet1d
    if ds == 'FMNIST_clean': # FMNIST has a simpler Unet1D
        model = UNet1D(
            input_channels=1,
            output_channels=1,
            base_channels=64,      # smaller for demonstration
            channel_mults=[1, 2],  # two resolution levels
            num_res_blocks=3,      # fewer blocks for simplicity
            dropout_rate=0.2,
            total_time_steps=training_config['unet_t']  # for the sinusoidal embedding

        ).to(device)
    else: 
        model = UNet1D(
            input_channels=1,
            output_channels=1,
            base_channels=64,      # smaller for demonstration
            channel_mults=[1, 2, 4],  # two resolution levels
            num_res_blocks=3,      # fewer blocks for simplicity
            dropout_rate=0.2,
            total_time_steps=training_config['unet_t'] ,  # for the sinusoidal embedding

        ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=training_config['lr'])
    ddim_sampler = DDIMSampler(model, num_timesteps=training_config['unet_t'],device=device)
    scaler = amp.GradScaler()

    # 4. Train the model

    save_path = REPO_HOME_DIR +f'records/latent_ddim/trained_models/{ds}-{version_num}/'
    latent_ddim_version_number = get_version_number(save_path)
    save_path = save_path + f"version_{latent_ddim_version_number}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print('ALREADY EXISTS:',save_path)


    train_ddim(model, 
               ddim_sampler,
               scaler,
               dataloader_train_pact, 
               training_config['batch_size'],
               optimizer, 
               training_config['unet_t'],
               device,
               training_config['num_epochs'],
               save_path=save_path)

    # 5.  Sampling testing
    model.eval()

    sampler = DDIMSampler(model, num_timesteps=training_config['unet_t'],device= device, eta=0.1)   
    num_gen_samples = 100
    num_proto = patronus_config_set['num_prototypes']
    samples_ddim = sampler.sample_ddim((num_gen_samples, 1, num_proto))
    print("Generated Samples DDIM:", samples_ddim)

    print('Training DDIM for latent sampling done. Model saved at:', save_path)

    return



def parse_args():
    parser = argparse.ArgumentParser(description='Train latent ddim')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--version_num', type=int, help='Version number of the trained model')
    parser.add_argument('--unet_t', type=int, default=100, help='Number of timesteps for the 1D UNet')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Training DDIM for latent sampling...')
    args = parse_args()
    ds = args.dataset
    version_num = args.version_num

    training_config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'unet_t': args.unet_t,
    }
    

    main(ds, version_num,
         training_config)
    
    
