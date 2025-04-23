
import pickle
import torch
import os

from models.patronus_unet import Patronus_Unet
from models.unet1d import UNet1D

from global_config import *



def get_checkpoint_dir(diff_train_from_ds:str,
                       version_num:int):
    dir_name = diff_train_from_ds if diff_train_from_ds.startswith('OUT') else diff_train_from_ds+'-train'
    return REPO_HOME_DIR+'/records/diffusion/patronus/'+dir_name+'/Logs_Checkpoints/checkpoints/version_{}/'.format(version_num)



def load_patronus_unet_model(ds_name: str,
                        version_num: int,
                        ):
    checkpoint_dir = get_checkpoint_dir(ds_name,version_num)

    # get the config
    with open(checkpoint_dir+'/config.pickle', 'rb') as handle:
        config_set = pickle.load(handle)

    print('^'*30+f'\nLoading trained DF model from {ds_name}')
    print(config_set)
    model = Patronus_Unet(
            # basics
            input_channels          = config_set['TrainingConfig']['IMG_SHAPE'][0],
            output_channels         = config_set['TrainingConfig']['IMG_SHAPE'][0],
            base_channels           = config_set['ModelConfig']['BASE_CH'],
            base_channels_multiples = config_set['ModelConfig']['BASE_CH_MULT'],
            apply_attention         = config_set['ModelConfig']['APPLY_ATTENTION'],
            dropout_rate            = config_set['ModelConfig']['DROPOUT_RATE'],
            time_multiple           = config_set['ModelConfig']['TIME_EMB_MULT'],
            # patronus
            num_proto               = config_set['TrainingConfig']['NUM_PROTOS'],
            prototype_vector_shape  = config_set['TrainingConfig']['P_VECTOR_SHAPE'],
            encoder_type            =config_set['TrainingConfig']['encoder_type'],
            img_size               = config_set['TrainingConfig']['IMG_SHAPE'],
            plb_channel_inputs     = config_set['TrainingConfig']['plb_channel_inputs'],
            plb_layer_filter_sizes = config_set['TrainingConfig']['plb_layer_filter_sizes'], 
            plb_layer_strides      = config_set['TrainingConfig']['plb_layer_strides'],
            plb_layer_paddings     = config_set['TrainingConfig']['plb_layer_paddings'], 
        )
    

    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])

    model.to(config_set['BaseConfig']['DEVICE'])
    
    model.eval()

    print('^'*30+f'\nSuccessful loaded model {ds_name} from {checkpoint_dir}.')
    return model, config_set

def load_latent_unet_model(ds_name:str,
                        version_num: int,
                        latent_version_num: int,
                        device:str = 'cuda'):

    checkpoint_path = os.path.join(REPO_HOME_DIR, f'records/latent_ddim/trained_models/{ds_name}-{version_num}/version_{latent_version_num}/best_model.pt')

    if ds_name == 'FMNIST_clean': 
        model = UNet1D(
            input_channels=1,
            output_channels=1,
            base_channels=64,      # smaller for demonstration
            channel_mults=[1, 2],  # two resolution levels
            num_res_blocks=2,      # fewer blocks for simplicity
            dropout_rate=0.1,
            total_time_steps=100,  # for the sinusoidal embedding

        ).to(device)
    
    else: 
        model = UNet1D(
            input_channels=1,
            output_channels=1,
            base_channels=64,      # smaller for demonstration
            channel_mults=[1, 2, 4],  # two resolution levels
            num_res_blocks=3,      # fewer blocks for simplicity
            dropout_rate=0.2,
            total_time_steps=100,  # for the sinusoidal embedding

        ).to(device)

    model.load_state_dict(torch.load(checkpoint_path))

    model.to(device)
    
    model.eval()

    print('^'*30+f'\nSuccessful loaded latent model {ds_name}-{version_num}.\n'+'^'*30)
    return model


def get(element: torch.Tensor, t: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, t)
    return ele.reshape(-1, 1, 1, 1)


def setup_log_directory(config):
    '''Log and Model checkpoint directory Setup'''
    
    if os.path.isdir(config.root_log_dir):
        # Get all folders numbers in the root_log_dir
        folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(config.root_log_dir)]
        
        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)

        # New version name
        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = config.log_dir

    # Update the training config default directory 
    log_dir        = os.path.join(config.root_log_dir,        version_name)
    checkpoint_dir = os.path.join(config.root_checkpoint_dir, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(log_dir,        exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")
    
    return log_dir, checkpoint_dir, version_name # add version name as the model output


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, dict):
        # Apply to_device recursively on dictionary items
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, str):
        return data
    else:
        return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)