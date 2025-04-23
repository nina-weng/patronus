import sys
sys.path.append('../patronus/')
from global_config import * # load REPO_HOME_DIR, DATASET_DIR


from train.train_config import BaseConfig,TrainingConfig,ModelConfig,save_config_train_diffusion
from train.utils import *
from train.dataloader import get_dataloader,inverse_transform


from models.patronus_unet import Patronus_Unet
from models.diffusion import SimpleDiffusion
from models.loss_fn import PrototypeDistinctLoss

import gc
import os
# from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import torch
import torch.nn as nn
from torch.cuda import amp
import torchvision.transforms as TF
from torchvision.utils import make_grid



from torchmetrics import MeanMetric
# from IPython.display import display
import wandb


# Algorithm 1: Training
def train_one_epoch(model, sd, loader, optimizer, scaler, 
                    loss_fn_list,
                    loss_lambda, 
                    epoch=800, 
                   base_config=BaseConfig(), training_config=TrainingConfig(),
                   log_interval=50,
                   validate_sameples_from_trainset = None,
                   **kwargs):
    
    loss_record = MeanMetric()

    # get the loss functions by loss_fn_list and loss_lambda
    loss_fn = {}
    for ind, loss_fn_name in enumerate(loss_fn_list):
        if loss_fn_name == 'denoiser':
            loss_fn['denoiser'] = {}
            loss_fn['denoiser']['func'] = nn.MSELoss()
            loss_fn['denoiser']['lambda'] = loss_lambda[ind]
        elif loss_fn_name == 'distinct':
            loss_fn['distinct'] = {}
            if kwargs.get("distinct_margin", False):
                distinct_margin = kwargs["distinct_margin"]
                loss_fn['distinct']['func'] = PrototypeDistinctLoss(distance_type="cosine",
                                                                    margin=distinct_margin)
            else: 
                loss_fn['distinct']['func'] = PrototypeDistinctLoss(distance_type="cosine")
            loss_fn['distinct']['lambda'] = loss_lambda[ind]
        else:
            raise ValueError('loss_fn_name not recognized : {}'.format(loss_fn_name))

    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")
         
        for ind, (x0s, _) in enumerate(loader):
            model.train()
            tq.update(1)
            
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = sd.forward_diffusion(x0s, ts)
            proact_x0s = model.get_proact(x0s)

            with amp.autocast():
                pred_noise = model(xts, ts, given_cond_vector=proact_x0s)
                learned_p = model.get_learned_prototypes()

                ###########
                # get loss
                this_loss = 0
                for loss_name in loss_fn_list:
                    if loss_name == 'denoiser':
                        denoiser_loss = loss_fn['denoiser']['func'](gt_noise, pred_noise)
                        this_loss += loss_fn['denoiser']['lambda'] * denoiser_loss
                    elif loss_name == 'distinct':
                        distinct_loss = loss_fn['distinct']['func'](learned_p) 
                        this_loss += loss_fn['distinct']['lambda'] * distinct_loss
                    else:
                        raise ValueError('loss_fn_name not recognized : {}'.format(loss_name))

                loss = this_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.6f}")
            if ind % log_interval == 0:
                wandb.log({"Loss-train-step": loss_value},step=(epoch-1)*len(loader)+ind)
                if len(loss_fn_list) >1:
                    wandb.log({"Loss-train-step-denoiser": denoiser_loss.item()},step=(epoch-1)*len(loader)+ind)
                    wandb.log({"Loss-train-step-distinct": distinct_loss.item()},step=(epoch-1)*len(loader)+ind)

            if ind == len(loader) - 1: # last batch 
                with torch.no_grad():
                    cond_vectors_train_last_patch = model.get_proact(validate_sameples_from_trainset)
                    generated_ = reverse_diffusion(model, 
                                                   sd, 
                                                   timesteps=training_config.TIMESTEPS, 
                                                   num_images=len(validate_sameples_from_trainset), 
                                                   img_shape=training_config.IMG_SHAPE,
                                                   device=base_config.DEVICE,
                                                   cond_vector=cond_vectors_train_last_patch,
                                                   isSave=False,
                                                   )
                    
                    # wandb the original image and also the generated image
                    assert validate_sameples_from_trainset.shape[0] == generated_.shape[0]
                    grid = make_grid(inverse_transform(validate_sameples_from_trainset).type(torch.uint8), nrow=8, pad_value=255.0).to("cpu")
                    pil_image_original = TF.functional.to_pil_image(grid)

                    grid = make_grid(inverse_transform(generated_).type(torch.uint8), nrow=8, pad_value=255.0).to("cpu")
                    pil_image_generated = TF.functional.to_pil_image(grid)


        mean_loss = loss_record.compute().item()
        
        wandb.log({"Loss-train-mean-loss-per-epoch": mean_loss,
                    # "Original Image": [wandb.Image(pil_image_original, caption="Original Image")],
                    "Generated Image": [wandb.Image(pil_image_generated, caption="Generated Image")]},
                  step=(epoch-1)*len(loader)+ind)
        
        # only log the original image at the first epoch
        if epoch == 1:
            wandb.log({"Original Image": [wandb.Image(pil_image_original, caption="Original Image")]},
                      step=(epoch-1)*len(loader)+ind)
    
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.6f}")

        
    torch.cuda.empty_cache()
    return mean_loss 

# Algorithm 2: Sampling
@torch.no_grad()
def reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64), 
                      num_images=5, nrow=8, device="cpu", 
                      cond_vector=None,
                      isSave = True, # whether save the image or not
                      **kwargs):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()



    for time_step in tqdm(iterable=reversed(range(1, timesteps)), 
                          total=timesteps-1, 
                          dynamic_ncols=False, 
                          desc="Sampling :: ", 
                          position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts, given_cond_vector=cond_vector)

        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts) 

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z # noise term
        )

            

    # Save the image at the final timestep of the reverse process. 
    if isSave:
        x = inverse_transform(x).type(torch.uint8)
        grid = make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = TF.functional.to_pil_image(grid)
        pil_image.save(kwargs['save_path'], format=save_path[-3:].upper())
    return x # notice that this x is after inverse_transform

    

def get_random_from_test_set(df_test, # should be the df of the test data
                             num_samples,
                             random_state=None):
        if random_state is not None:
            random_samples = df_test.sample(n=num_samples, random_state=random_state)
        else:
            random_samples = df_test.sample(n=num_samples)
        # get the image paths
        random_samples_img = random_samples['img'].values
        # get the labels
        random_samples_lab = random_samples['label'].values
        return random_samples_img, random_samples_lab





if __name__ == '__main__':
    wandb.init(project="patronus",)   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    print('*'*30)
    print('PATRONUS')
    print('*'*30)

    # set up the log/checkpoint directory
    log_dir, checkpoint_dir, version_name = setup_log_directory(config=BaseConfig())
    print(f'{log_dir=}')

    # save the config
    save_config_train_diffusion(TrainingConfig(),ModelConfig(),BaseConfig(),save_dir = checkpoint_dir)


    # load the data 
    dataloader_train = get_dataloader(dataset_name=BaseConfig.ds_name+'-train', 
                                      batch_size=TrainingConfig.BATCH_SIZE,
                                      device=device,
                                      shuffle=True)
    dataloader_test = get_dataloader(dataset_name=BaseConfig.ds_name+'-test', 
                                     batch_size=TrainingConfig.BATCH_SIZE,
                                     device=device,
                                     shuffle=False)

    # get the test samples in df
    test_data = []

    # Iterate through the test DataLoader
    for batch in tqdm(dataloader_test):
        imgs, extra_info = batch  # adjust based on your DataLoader output format
        labs, img_ids = extra_info  

        # Iterate through each item in the batch
        for img, lab in zip(imgs, labs):    
            img = img.cpu().detach()
            lab = lab.cpu().detach()     
            # Append the data to the list

            if isinstance(lab, torch.Tensor):
                if lab.numel() == 1:  # Scalar tensor
                    label = lab.item()
                else:  # Tensor with multiple elements
                    label = lab.tolist()  # Convert to a Python list
            else:
                raise ValueError("Unexpected type for 'lab': {}".format(type(lab)))

            test_data.append({
                "img": img,
                "label": label  # convert label tensor to a scalar
            })


    # Convert the list of dictionaries to a DataFrame
    df_test = pd.DataFrame(test_data) 

    # get samples for inference time inspectation
    num_samples = 64
    sample_imgs, sample_labs = get_random_from_test_set(df_test,
                                    num_samples=num_samples, 
                                    random_state = 2024)
    # save the original test samples
    sample_imgs_4plot = [inverse_transform(img).type(torch.uint8) for img in sample_imgs]
    # convert the images to a grid
    grid = make_grid(sample_imgs_4plot, nrow=8, pad_value=255.0).to("cpu")
    pil_image = TF.functional.to_pil_image(grid)
    pil_image.save(log_dir +f"/original.png", format='PNG')

    # get the sample tensor for inference at each epoch
    sample_imgs_arry = [np.array(vec, dtype=np.float32) for vec in sample_imgs]
    sample_imgs_tensor = torch.tensor(np.stack(sample_imgs_arry), dtype=torch.float32).to(device)
    # print(f'{sample_imgs_tensor.shape=}')

   

    print('*'*30 + 'Load model' + '*'*30)


    model = Patronus_Unet(
        # basics
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
        # patronus related
        num_proto               = TrainingConfig.NUM_PROTOS,
        prototype_vector_shape  = TrainingConfig.P_VECTOR_SHAPE,
        encoder_type            = TrainingConfig.encoder_type,
        img_size                = TrainingConfig.IMG_SHAPE,

        plb_channel_inputs      = TrainingConfig.plb_channel_inputs,
        plb_layer_filter_sizes  = TrainingConfig.plb_layer_filter_sizes,
        plb_layer_strides       = TrainingConfig.plb_layer_strides,
        plb_layer_paddings      = TrainingConfig.plb_layer_paddings,
    )
    model.to(BaseConfig.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

    loss_fn_list = TrainingConfig.loss_fn
    loss_lambda = TrainingConfig.loss_lambda

    sd = SimpleDiffusion(
            num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
            img_shape               = TrainingConfig.IMG_SHAPE,
            device                  = BaseConfig.DEVICE,
        )

    scaler = amp.GradScaler()


    # -------

    # add noise and save the noisy images -- for validating the noise scheduling
    print('*'*30 + 'Add noise (Validating noise scheduling)' + '*'*30)
    noisy_images = []
    sample_imgs_tensor_randon_ind = torch.randint(low=0, high=sample_imgs_tensor.shape[0], size=(6,))# randomly select 6 images
    if TrainingConfig.TIMESTEPS == 1000:
        specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]
    elif TrainingConfig.TIMESTEPS == 200:
        specific_timesteps = [0, 5, 10, 25, 50, 100, 150, 199]
    else:
        raise Exception('Not implemented timesteps')

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long,device=device)

        xts, _ = sd.forward_diffusion(sample_imgs_tensor[sample_imgs_tensor_randon_ind], timestep)
        xts = inverse_transform(xts) / 255.0
        xts = make_grid(xts, nrow=1, padding=1)

        noisy_images.append(xts)

    # Plot and see samples at different timesteps

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor="white")

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        noisy_sample = noisy_sample.cpu().detach()
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(os.path.join(log_dir,'forward_diffusion_process_example.png'))



    # ------------ 

    # load the model from the previous training if not new training
    if TrainingConfig.new_training == False: 
        training_from_version = TrainingConfig.training_from_version
        # load the model
        model_from_dir = REPO_HOME_DIR + '/records/diffusion/patronus/{}/Logs_Checkpoints/checkpoints/version_{}'.format(BaseConfig.DATASET,
                                                                                                                            training_from_version)
        checkpoint_dict = torch.load(os.path.join(model_from_dir, f"ckpt.tar"))
        model.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["opt"])
        scaler.load_state_dict(checkpoint_dict["scaler"])
        del checkpoint_dict

    total_epochs = TrainingConfig.NUM_EPOCHS + 1


    # training
    print('^'*30+'START Training' + '^'*30)

    print(f'Device: {BaseConfig.DEVICE}')



    # get the random 16 images from train set for wandb inspection at validation step
    num_selected_train_samples = 16
    for ind, (x0s, _) in enumerate(dataloader_train):
        x0s_val = x0s[:num_selected_train_samples]
        break



    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        
        train_one_epoch(model, sd, dataloader_train, optimizer, scaler, 
                        loss_fn_list= loss_fn_list,
                        loss_lambda = loss_lambda,
                        epoch=epoch,
                        validate_sameples_from_trainset=x0s_val,
                        log_interval=TrainingConfig.LOG_INTERVAL,
                        distinct_margin=TrainingConfig.distinct_margin,
                        )



        if epoch % 5 == 0 or epoch == 1:
            print(f'Sampling at {epoch=}')
            save_path = os.path.join(log_dir, f"{epoch}.png")


            # Algorithm 2: Sampling
            # inference time so stop the training
            model.eval()
            with torch.no_grad():
                # get the condition p from the selelcted test samples
                cond_vectors = model.get_proact(sample_imgs_tensor)

                reverse_diffusion(model, 
                                  sd, 
                                  timesteps=TrainingConfig.TIMESTEPS, 
                                  num_images=num_samples, 
                                  save_path=save_path, 
                                  img_shape=TrainingConfig.IMG_SHAPE, 
                                  device=BaseConfig.DEVICE,
                                  cond_vector=cond_vectors,
                )

                # clear_output()
                checkpoint_dict = {
                    "opt": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "model": model.state_dict()
                }
                torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
                del checkpoint_dict


    wandb.finish()



