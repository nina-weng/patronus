'''
Conditional sampling. 
1. Getting the prototype activations of the testset
2. sample the images with those activations
3. save those generated images

how to use, e.g. 
`python ./train/conditional_sampling.py --dataset CelebA --version_num 4`
'''

import torch
from PIL import Image
from tqdm import tqdm

import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


import sys
sys.path.append('../patronus/')
# print(sys.path)
from global_config import * # load REPO_HOME_DIR, DATASET_DIR
from train.utils import load_patronus_unet_model
from train.dataloader import get_dataloader_pact,inverse_transform
from models.diffusion import SimpleDiffusion



def parse_args():
    parser = argparse.ArgumentParser(description='Conditional sampling')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--version_num', type=int, help='Version number of the trained model')
    parser.add_argument('--total_gen_num', type=int, default=10000, help='Total number of generated images')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    # parameters and settings
    ds = args.dataset
    version_num = args.version_num
    total_gen_num = args.total_gen_num

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # where to save the generated images
    save_dir = REPO_HOME_DIR+ f'/records/FID/cond_{ds}_{version_num}/'
    gen_save_dir = save_dir + 'gen_imgs/'

    # generate the images
    if not os.path.exists(gen_save_dir):
        os.makedirs(gen_save_dir)


    # save multiple runs
    def get_run_version_num(gen_save_dir):
        run_version_num = 1
        while os.path.exists(gen_save_dir + 'run_'+str(run_version_num)+'/'):
            run_version_num += 1
        return run_version_num
    run_version_num = get_run_version_num(gen_save_dir)
    gen_save_dir = gen_save_dir + 'run_'+str(run_version_num)+'/'
    os.makedirs(gen_save_dir)
    print(f'save generate images to: {gen_save_dir}')

    # load model
    print('*'*30 + 'Load model' + '*'*30)
    model, patronus_config_set = load_patronus_unet_model(ds_name=ds, 
                                                        version_num=version_num,
                                )
    
    model.to(device)
    model.eval()
    prototype_encoder = model.proactBlock
    prototype_encoder.eval()

    # load test set
    # get the prototype activation for the training data and wrap it as dataloader
    dataloader_test_pact = get_dataloader_pact(dataset_name=f'{ds}-test',
        batch_size=64,
        pact_encoder=prototype_encoder,
        device=device,
        shuffle = False,  # do not need to shuffle here
    )

    # diffusion scheduler
    sd = SimpleDiffusion(
        num_diffusion_timesteps = patronus_config_set['TrainingConfig']['TIMESTEPS'],
        img_shape               = patronus_config_set['TrainingConfig']['IMG_SHAPE'],
        device                  = patronus_config_set['BaseConfig']['DEVICE'],
    )
    
    cnt=0

    for i, (x, _) in tqdm(enumerate(dataloader_test_pact)):
        print(f'generating images from batch {i+1}/{len(dataloader_test_pact)}')
        with torch.no_grad():
            latent_samples = x
            latent_samples = latent_samples.squeeze(1)
            bs = latent_samples.shape[0]
  
            # generate the images    
            gen_imgs = sd.sample(model,
                    shape=(bs,) +  patronus_config_set['TrainingConfig']['IMG_SHAPE'],
                    noise=None,
                    progress=True,
                    model_kwargs={'given_cond_vector':latent_samples},
                    num_samples=1,
                    eta=0.0)
        
                                    
            for j in tqdm(range(bs)):
                idx = i * bs + j
                img = inverse_transform(gen_imgs[j]).cpu().numpy()  # Remove permute for flexibility
                
                # Handle single-channel images
                if img.shape[0] == 1:  # If single-channel (1, H, W)
                    img = img.squeeze(0)  # Remove channel dimension (H, W)
                else:  # For RGB images (3, H, W)
                    img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)
                
                img = (img.clip(0, 255)).astype(np.uint8)
                
                img_pil = Image.fromarray(img)
                img_pil.save(os.path.join(gen_save_dir, f"{idx:05d}.jpg"), "JPEG")
                
                if idx >= total_gen_num:
                    break

        

            cnt+=bs
            print(f'generated {cnt} images')
            if cnt > total_gen_num:
                break
    
    if cnt < total_gen_num:
        print(f'Warning: only {cnt} images are generated, less than the required {total_gen_num}')

    print(f'All generated images are saved to {gen_save_dir}')





    

