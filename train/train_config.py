from train.utils import get_default_device

from dataclasses import dataclass,asdict,field
import os
import pprint
import pickle

from global_config import *

from typing import List, Tuple




@dataclass
class BaseConfig:
    repo_home_dir: str = REPO_HOME_DIR
    DEVICE : str = get_default_device()
    exp_type: str = 'patronus' 
    ds_name : str = 'CelebA_hair_smile'  ### <--- change this to the dataset you want to train on
    
    # -------- ds_name could be chose from ---------
    # ['MNIST_clean','CelebA','FMNIST_clean','cifar10','ffhq256', 'CHEXPERT_CardioPM', 'OUT_CHEXPERT'] 
    # manipulated one: ['CelebA_hair_smile']

    split: str ='train' # 'train', 'test'
    record_dir: str = './records/diffusion/'+exp_type+'/'
    DATASET: str = ds_name+'-'+split 
    

    # For logging inferece images and saving checkpoints.
    root_log_dir: str = os.path.join(record_dir,DATASET,"Logs_Checkpoints", "Inference")
    root_checkpoint_dir:str = os.path.join(record_dir,DATASET,"Logs_Checkpoints", "checkpoints")


    # Current log and checkpoint directory.
    log_dir:str = "version_0"
    checkpoint_dir:str = "version_0"



@dataclass
class TrainingConfig:
    TIMESTEPS: int = 1000 # Define number of diffusion timesteps
    IMG_SHAPE: tuple[int]
    BATCH_SIZE: int
    SAMPLE_NUM: int
    NUM_EPOCHS: int

    # patronus related
    NUM_PROTOS: int
    P_VECTOR_SHAPE: tuple[int] = (1,128)
    dataset: str = BaseConfig.DATASET
    encoder_type: str = 'basic_conv'
    new_training: bool = False   # < -- train from scratch; if needed, continue training from a previous version
    training_from_version: int = 1 # <-- change this to the version you want to continue training from
    loss_fn:  Tuple[str, ...] = ('denoiser','distinct')
    loss_lambda : tuple[float] = (1.0,0.0)
    distinct_margin: float = 0.2

    plb_channel_inputs: tuple[int]
    plb_layer_filter_sizes: tuple[int]
    plb_layer_strides: tuple[int]
    plb_layer_paddings: tuple[int]


    if ('CHEXPERT' in BaseConfig.DATASET):
        IMG_SHAPE = (1,224,224)
        BATCH_SIZE = 16 
        SAMPLE_NUM = min(BATCH_SIZE*4,32)
        NUM_EPOCHS = 300
        LOG_INTERVAL = 50
        NUM_PROTOS = 100
        
    elif 'MNIST' in BaseConfig.DATASET: # includes both ['MNIST_clean', 'FMNIST_clean'] 
        IMG_SHAPE = (1,32,32) 
        BATCH_SIZE = 128
        SAMPLE_NUM = min(BATCH_SIZE*4,32)
        NUM_EPOCHS = 100
        LOG_INTERVAL = 50
        NUM_PROTOS = 30
    
    elif 'cifar10' in BaseConfig.DATASET:
        IMG_SHAPE = (3,32,32)
        BATCH_SIZE = 128
        SAMPLE_NUM = min(BATCH_SIZE*4,32)
        NUM_EPOCHS = 300
        LOG_INTERVAL = 50
        NUM_PROTOS = 100
    
    elif ('CelebA' in BaseConfig.DATASET) or ('ffhq256' in BaseConfig.DATASET):
        IMG_SHAPE = (3,64,64)
        BATCH_SIZE = 64 
        SAMPLE_NUM = min(BATCH_SIZE*4,32)
        NUM_EPOCHS = 300
        LOG_INTERVAL = 50
        NUM_PROTOS = 100

    else:
        raise Exception(f'Not implemented {BaseConfig.DATASET=}')
    LR : float = 1e-5
    NUM_WORKERS : int = 2

    if 'CHEXPERT' in BaseConfig.DATASET:
        plb_channel_inputs = (1, 32, 64, 64, 128, 128)
        plb_layer_filter_sizes = (5,5,5,5,5)
        plb_layer_strides = (2,2,1,1,1)
        plb_layer_paddings = (1,1,0,0,0)
    elif 'CelebA' in BaseConfig.DATASET or 'ffhq256' in BaseConfig.DATASET \
        or 'MNIST' in BaseConfig.DATASET or 'cifar10' in BaseConfig.DATASET:
        plb_channel_inputs= (1, 32, 64, 64, 128)
        plb_layer_filter_sizes = (3,3,3,3)
        plb_layer_strides = (2,1,1,1)
        plb_layer_paddings = (1,0,0,0)
    else:
        raise Exception(f'Not implemented {BaseConfig.DATASET=}')
        



@dataclass
class ModelConfig:
    BASE_CH: int = 64  
    BASE_CH_MULT: tuple[int] = (1, 2, 4, 4) 
    APPLY_ATTENTION: tuple[bool] = (False, False, False, False)
    ATTENTION_TYPE: str= 'ori_att'
    DROPOUT_RATE : float = 0.1
    TIME_EMB_MULT : int = 4 





def save_config_train_diffusion(tc: TrainingConfig,
                                mc: ModelConfig,
                                bc: BaseConfig,
                                save_dir: str):
    '''
    save the config of diffusion model trainingS
    '''
    # save as dict
    config_dict = {}
    config_dict['BaseConfig'] = asdict(bc)
    config_dict['ModelConfig'] = asdict(mc)
    config_dict['TrainingConfig'] = asdict(tc)

    with open(save_dir+'/config.pickle', 'wb') as handle:
        pickle.dump(config_dict, handle)

    # save to txt
    config_str = pprint.pformat(config_dict,sort_dicts=False)
    with open(save_dir+'/config.txt', "w") as f:
        f.write(config_str)

    return

