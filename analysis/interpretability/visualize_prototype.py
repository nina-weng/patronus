'''
Visulize the learned prototypes from trained patronus.
Descriptive steps are introduced in Sec. 3.3 in the paper. 
run: 
python3 analysis/interpretability/visualize_prototype.py
'''

import sys
sys.path.append('../patronus/')
from global_config import * # load REPO_HOME_DIR, DATASET_DIR


import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from train.utils import load_patronus_unet_model
from train.dataloader import get_dataloader, inverse_transform
from models.diffusion import SimpleDiffusion
from analysis.interpretability.visualization_config import VisConfig

def plot_vis_p(highest_activated_record,
               patronus_config_set,
               isSave=False,
               save_path=None,
               choose_p_ind: list = None,
               with_s=False,
               fig_height=3,
               swap_axes=False,
               fontsize = 30,
               ):  # New option to swap row/column layout
    '''
    Flexibly plot the activated patch with given p_ind.

    Parameters:
    - highest_activated_record: Dictionary containing image records.
    - isSave: Whether to save the figure.
    - save_path: Path to save the figure.
    - choose_p_ind: List of prototype indices to plot.
    - with_s: Whether to display activation scores.
    - fig_height: Height of the figure.
    - swap_axes: If True, swaps rows and columns in the subplot layout.
    - fontsize: fontsize for extra information, by default = 20
    '''
    
    assert choose_p_ind is not None
    assert all(element in highest_activated_record.keys() for element in choose_p_ind)
    
    p_i_key = sorted(choose_p_ind)

    original_img = torch.stack([highest_activated_record[p_i]['ori_img'] for p_i in p_i_key], dim=0)
    generated_pact_img = torch.stack([highest_activated_record[p_i]['enhanced_img'] for p_i in p_i_key], dim=0)
    most_activated_patches = torch.stack([highest_activated_record[p_i]['enhanced_patch'] for p_i in p_i_key], dim=0)
    bounding_boxes = [highest_activated_record[p_i]['enhanced_b_box'] for p_i in p_i_key]
    pro_act_generated_img = torch.stack([highest_activated_record[p_i]['enhanced_p_act'] for p_i in p_i_key], dim=0)
    pro_act_generated_img_ori = torch.stack([highest_activated_record[p_i]['ori_p_act'] for p_i in p_i_key], dim=0)

    num_prototypes = len(p_i_key)
    num_rows, num_cols = (num_prototypes, 3) if swap_axes else (3, num_prototypes)

    fig_width = fig_height * num_cols if not swap_axes else fig_height * num_rows
    fig_height = fig_height * num_rows if not swap_axes else fig_height * num_cols

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), dpi=50)

    if num_prototypes == 1:
        ax = np.expand_dims(ax, axis=0 if swap_axes else 1)  # Keep array structure consistent

    def get_axes(row, col):
        """Helper function to access the correct subplot based on swap_axes"""
        return ax[col, row] if swap_axes else ax[row, col]

    # Plot original images
    for i in range(num_prototypes):
        this_oimg = original_img[i]
        img_ax = get_axes(0, i)
        if patronus_config_set['TrainingConfig']['IMG_SHAPE'][0] == 1:
            img_ax.imshow(this_oimg.cpu().squeeze().numpy(), cmap='gray')
        else:
            img = np.transpose(inverse_transform(this_oimg).type(torch.uint8).cpu().squeeze().numpy(), (1, 2, 0))
            img_ax.imshow(img)
        if with_s:
            img_ax.set_title(f'{pro_act_generated_img_ori[i]:.2f}', fontsize=fontsize)
        img_ax.axis('off')

    # Plot generated images with bounding boxes
    for i in range(num_prototypes):
        img = generated_pact_img[i]
        img_ax = get_axes(1, i)
        if patronus_config_set['TrainingConfig']['IMG_SHAPE'][0] == 1:
            img_ax.imshow(img.cpu().squeeze().numpy(), cmap='gray')
        else:
            img_trans = np.transpose(inverse_transform(img).type(torch.uint8).cpu().squeeze().numpy(), (1, 2, 0))
            img_ax.imshow(img_trans)

        # Plot bounding boxes
        b_box = bounding_boxes[i]
        lines = [
            ([b_box[0], b_box[2]], [b_box[1], b_box[1]]),
            ([b_box[0], b_box[2]], [b_box[3], b_box[3]]),
            ([b_box[0], b_box[0]], [b_box[1], b_box[3]]),
            ([b_box[2], b_box[2]], [b_box[1], b_box[3]])
        ]
        for line in lines:
            img_ax.plot(*line, 'r',linewidth=5)

        if with_s:
            img_ax.set_title(f'{pro_act_generated_img[i]:.2f}', fontsize=fontsize)
        img_ax.set_xlim([0, img.shape[1]])
        img_ax.set_ylim([img.shape[2], 0])
        img_ax.axis('off')

    # Plot most activated patches
    for i in range(num_prototypes):
        p = most_activated_patches[i].squeeze(0)
        img_ax = get_axes(2, i)
        if patronus_config_set['TrainingConfig']['IMG_SHAPE'][0] == 1:
            img_ax.imshow(p.detach().cpu().numpy().transpose(1, 2, 0), cmap='gray')
        else:
            img_trans = np.transpose(inverse_transform(p).type(torch.uint8).cpu().squeeze().numpy(), (1, 2, 0))
            img_ax.imshow(img_trans)
        img_ax.set_title(f'vis p_{p_i_key[i]}', fontsize=fontsize)
        img_ax.axis('off')

    plt.axis('off')
    plt.tight_layout(h_pad=0.0)

    if isSave:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()
    else: print('Not saving the result, you can adjust this code in notebook and show it.')


def get_bounding_box_from_internal_layer(i,j,patch_size):
    # TODO
    if patch_size == 14:
        b_box = [i*2, j*2,i*2+patch_size,j*2+patch_size]
        b_box = [each.detach().cpu().item() for each in b_box]
    else:
        raise NotImplementedError
    return b_box


def get_most_activated_patch_for_one(generated_pact_img, # generated enhanced images
                                     p_ind, # prototype index, a list
                                     model, # UNet,
                             ):
    '''
    get the most activated patch for one img, given one or more prototype index
    '''
    assert generated_pact_img.shape[0] == 1
    most_activated_patches = []
    bounding_boxes = []
    num_patches = model.proactBlock.proto_layer_rf_info[0]**2
    patch_size = model.proactBlock.proto_layer_rf_info[2]-1   #TODO: compability
    for p_i in p_ind:
        hidden_rep = model.proactBlock.encoder(generated_pact_img) # (BS, channel_inputs[-1], hidden_rep_size, hidden_rep_size)
        hidden_rep_size = hidden_rep.shape[2]

        k2 = hidden_rep.permute(0, 2, 3, 1) # (BS, hidden_rep_size, hidden_rep_size, channel_inputs[-1])
        k2 = k2.view(k2.shape[0], hidden_rep_size*hidden_rep_size,model.proactBlock.channel_inputs[-1]).detach() # (BS, hidden_rep_size*hidden_rep_size, channel_inputs[-1])

        distances = model.proactBlock.prototype_distances(hidden_rep) # distances.shape=torch.Size([BS, num_p, 2, 2]) 

        neg_distances = -distances
        min_distances,min_indices = F.max_pool2d(neg_distances,
                                            kernel_size=(distances.size()[2],
                                                        distances.size()[3]),
                                                        return_indices=True)
        min_distances = -min_distances # distances.shape=torch.Size([BS, num_p]) 
        min_indices = min_indices.view(min_indices.shape[0], -1) # (BS, num_prototypes)

        this_p_indice = min_indices[0,p_i]
        
        sqrt_num_patches = int(np.sqrt(num_patches))

        min_j = this_p_indice // sqrt_num_patches
        min_i = this_p_indice % sqrt_num_patches

        this_bounding_box = get_bounding_box_from_internal_layer(min_i,min_j,patch_size=patch_size)

        this_patch = generated_pact_img[:,:,this_bounding_box[0]:this_bounding_box[2],this_bounding_box[1]:this_bounding_box[3]]
        this_patch = generated_pact_img[:,:,this_bounding_box[1]:this_bounding_box[3],this_bounding_box[0]:this_bounding_box[2]] 

        most_activated_patches.append(this_patch)
        bounding_boxes.append(this_bounding_box)

    return most_activated_patches,bounding_boxes



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # the result will save to
    save_dir = REPO_HOME_DIR + 'records/vis_prototype/'
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    # ---- Load the patronus model -----
    print('*'*30 + 'Load model' + '*'*30)
    model, patronus_config_set = load_patronus_unet_model(ds_name=VisConfig.ds_name, 
                                                        version_num=VisConfig.version_num,
                                )


    # ---- Load the test set ----- 
    # we use test set to avoid memorization from the training set
    dataloader_test = get_dataloader(
        dataset_name=f'{VisConfig.ds_name}-test',
        batch_size=256,
        device='cpu',
        shuffle = True, 
    )

    # ---- Get the plausible maximum similarity score: max(sX) ----
    for b_image, extra_info in dataloader_test:
        # print(b_image.shape)
        # b_img_id = extra_info[1]
        pact = model.proactBlock(b_image.to(device))
        max_pact = torch.max(pact).detach().cpu()
        tail85_pact = torch.quantile(pact, 0.85).detach().cpu()
        tail15_pact = torch.quantile(pact, 0.15).detach().cpu()
        tail95_pact = torch.quantile(pact, 0.95).detach().cpu()
        print(f'{max_pact=}\n{tail85_pact=}\n{tail15_pact=}\n{tail95_pact=}\n')
        break

    # ---- Load the diffusion schduler ----
    sd = SimpleDiffusion(
            num_diffusion_timesteps = patronus_config_set['TrainingConfig']['TIMESTEPS'],
            img_shape               = patronus_config_set['TrainingConfig']['IMG_SHAPE'],
            device                  = patronus_config_set['BaseConfig']['DEVICE'],
    )


    # ---- get the randomly selected n samples, will get the most activated patch from them ---- 
    batch_size = b_image.shape[0]
    if VisConfig.seed is not None: torch.manual_seed(VisConfig.seed)
    random_pick_samples_ind = torch.randperm(batch_size,)[:VisConfig.num_random_pick_samples]

    random_pick_pact = pact[random_pick_samples_ind]
    random_pick_img = b_image[random_pick_samples_ind]

    # --- get x_T ----- 
    random_pick_xT = sd.reverse_sample_loop(model,random_pick_img.to(device), model_kwargs={'given_cond_vector':random_pick_pact})['sample']

    # for each selected p, maximum it's possible value, and then generate the image
    highest_activated_record = {}
    enhanced_pact_chosen_p_all = []
    selected_p_ind = list(VisConfig.vis_p_ind)

    for real_i,i_p in enumerate(selected_p_ind):
        # deep copy
        enhanced_pact_chosen_p = random_pick_pact.clone()
        enhanced_pact_chosen_p[:,i_p] = max_pact.item()
        enhanced_pact_chosen_p_all.append(enhanced_pact_chosen_p)

    enhanced_pact_chosen_p_all = torch.cat(enhanced_pact_chosen_p_all,dim=0)

    random_pick_img_all = random_pick_img.repeat(len(selected_p_ind),1,1,1)
    random_pick_xT_all = random_pick_xT.repeat(len(selected_p_ind),1,1,1)

    this_x_0_enhanced_all = sd.sample(model,
                                        shape=random_pick_img_all.shape ,
                                        noise=random_pick_xT_all,
                                        progress=True,
                                        model_kwargs={'given_cond_vector':enhanced_pact_chosen_p_all},
                                        num_samples=1)
    

    highest_activated_record = {}
    num_selected_p =len(selected_p_ind)
    for real_i,i_p in enumerate(selected_p_ind):
        this_x_0_enhanced = this_x_0_enhanced_all[real_i*VisConfig.num_random_pick_samples:(real_i+1)*VisConfig.num_random_pick_samples]
        this_p_act_all = model.proactBlock(this_x_0_enhanced)
        this_p_act = this_p_act_all[:,i_p]

        this_p_act_all_original = model.proactBlock(random_pick_img.to(device))
        this_p_act_ori = this_p_act_all_original[:,i_p]

        # highest_ind = torch.argmax(this_p_act-this_p_act_ori)
        # another way:
        highest_ind = torch.argmax(this_p_act)

        # get the most activated patch for this image
        most_activated_patches_this,bounding_boxes_this = get_most_activated_patch_for_one(this_x_0_enhanced[highest_ind].unsqueeze(0),
                                                                    [i_p],
                                                                    model,)

        this_p_act_nor_softmax = torch.nn.functional.softmax(this_p_act_all, dim=1)

        highest_activated_record[i_p] = {'ori_img': random_pick_img[highest_ind],
                                        'enhanced_img':this_x_0_enhanced[highest_ind],
                                        'enhanced_patch':most_activated_patches_this[0],
                                        'enhanced_b_box':bounding_boxes_this[0],
                                        'enhanced_p_act':this_p_act[highest_ind],
                                        'enhanced_p_act_nor':this_p_act_nor_softmax[highest_ind,i_p],
                                        'ori_p_act':this_p_act_ori[highest_ind],
                                        }
        

    # ---- plot ---
    p_ind_str = '_'.join([str(i) for i in selected_p_ind])
    save_path = save_dir + f'{VisConfig.ds_name}_v{VisConfig.version_num}_numrandompick{VisConfig.num_random_pick_samples}_{p_ind_str}.pdf'
    plot_vis_p(highest_activated_record,
               patronus_config_set,
               choose_p_ind=selected_p_ind,
               isSave=True,
               save_path=save_path,
               with_s=True,
               fig_height = 4,
            # swap_axes=False,
            )


