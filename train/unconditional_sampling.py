'''
unconditional sampling
1. sample from latent ddim
2. sample the image from the difffusion unet model
3. save the generated images

e.g. `python ./train/unconditional_sampling.py --dataset CelebA --version_num 4 --latent_version 1`
'''

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import argparse

import sys
sys.path.append('../patronus/')
from global_config import *
from train.utils import load_latent_unet_model,load_patronus_unet_model
from train.dataloader import inverse_transform
from train.train_latent_ddim import DDIMSampler
from models.diffusion import SimpleDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description='Conditional sampling')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--version_num', type=int, help='Version number of the trained model')
    parser.add_argument('--total_gen_num', type=int, default=10000, help='Total number of generated images')
    parser.add_argument('--latent_version', type=int, default=0, help='Version number of the latent model')
    parser.add_argument('--eta', type=float, default=0.0, help='eta for latent sampling')
    parser.add_argument('--latent_sample_method', type=str, default='ddim', help='latent sampling method')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    ds = args.dataset
    version_num = args.version_num
    total_gen_num = args.total_gen_num
    latent_version = args.latent_version
    latent_sample_method = args.latent_sample_method
    eta = args.eta

    save_dir = REPO_HOME_DIR+ f'records/FID/uncond_{ds}_{version_num}_{latent_version}/'
    gen_save_dir = save_dir + 'gen_imgs/'

    # generate the images
    if not os.path.exists(gen_save_dir):
        os.makedirs(gen_save_dir)


    def get_run_version_num(gen_save_dir):
        run_version_num = 1
        while os.path.exists(gen_save_dir + 'run_'+str(run_version_num)+'/'):
            run_version_num += 1
        return run_version_num
    run_version_num = get_run_version_num(gen_save_dir)
    gen_save_dir = gen_save_dir + 'run_'+str(run_version_num)+'/'
    os.makedirs(gen_save_dir)
    print(f'save generate images to: {gen_save_dir}')

    # load latent ddim and protego
    latent_unet = load_latent_unet_model(ds_name=ds, 
                                    version_num=version_num, 
                                    latent_version_num= latent_version,
                                    device=device)
    print('successfully loaded latent unet model')

    # load model
    print('*'*30 + 'Load model' + '*'*30)
    model, patronus_config_set = load_patronus_unet_model(ds_name=ds, 
                                                        version_num=version_num,
                                )

    model.to(device)
    model.eval()
    num_proto = patronus_config_set['TrainingConfig']['NUM_PROTOS']

    sampler = DDIMSampler(latent_unet, num_timesteps=100,device= device, eta=eta) 

    sd = SimpleDiffusion(
        num_diffusion_timesteps = patronus_config_set['TrainingConfig']['TIMESTEPS'],
        img_shape               = patronus_config_set['TrainingConfig']['IMG_SHAPE'],
        device                  = patronus_config_set['BaseConfig']['DEVICE'],
    )


    # generate the images by 100 per batch
    samples_per_batch = 100
    num_batches = total_gen_num // samples_per_batch

    for i in range(num_batches):
        print(f'generating images from batch {i+1}/{num_batches}')
        with torch.no_grad():
            if latent_sample_method == 'ddim':
                latent_samples = sampler.sample_ddim((samples_per_batch, 1, num_proto))
            elif latent_sample_method == 'ddpm':
                latent_samples = sampler.sample_ddpm((samples_per_batch, 1, num_proto))
            else:
                raise NotImplementedError
            latent_samples = latent_samples.squeeze(1)

            # generate the images    
            gen_imgs = sd.sample(model,
                    shape=(samples_per_batch,) +  patronus_config_set['TrainingConfig']['IMG_SHAPE'],
                    noise=None,
                    progress=True,
                    model_kwargs={'given_cond_vector':latent_samples},
                    num_samples=1,
                    eta=0.0)
            
        
               
            for j in tqdm(range(samples_per_batch)):
                idx = i * samples_per_batch + j
                img = inverse_transform(gen_imgs[j]).cpu().numpy()  # Remove permute for flexibility
                
                # Handle single-channel images
                if img.shape[0] == 1:  # If single-channel (1, H, W)
                    img = img.squeeze(0)  # Remove channel dimension (H, W)
                else:  # For RGB images (3, H, W)
                    img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)
                
                img = (img.clip(0, 255)).astype(np.uint8)
                
                img_pil = Image.fromarray(img)
                img_pil.save(os.path.join(gen_save_dir, f"{idx:05d}.jpg"), "JPEG")

    
    print(f'Finished generating {total_gen_num} images from {ds} dataset.')


