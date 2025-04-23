from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import pandas as pd


from train.train_config import TrainingConfig,BaseConfig
from train.utils import DeviceDataLoader
from global_config import *
from train.transform_config import *




class CustomMetaDataset(Dataset):
    def __init__(self, ds_name, meta_file, data_root, transform=None, split="train",
                 encoder_func=None, device='cpu'):
        """
        Dataset using a metadata file to load train/val/test splits and class labels.

        Args:
            meta_file (str): Path to the CSV file containing metadata (e.g., paths and splits).
            data_root (str): Root directory of the image data.
            transform (callable, optional): Transform to apply to the images.
            split (str): One of 'train', 'val', or 'test' to filter the split.
        """
        self.meta_file = meta_file
        self.data_root = data_root
        self.transform = transform
        self.split = split
        self.ds_name = ds_name  
        self.encoder_func = encoder_func 
        self.device = device   

        # Load metadata
        self.meta_df = pd.read_csv(meta_file)


        if 'CelebA' in self.ds_name:
            label_name = 'all' 
            all_y_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                   'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                   'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                   'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                   'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                   'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
                   ]
            # Filter by split
            split = 0 if split == 'train' else (1 if split == 'val' else 2)
            self.meta_df = self.meta_df[self.meta_df["split"] == split].reset_index(drop=True)  

        elif 'ffhq256' in self.ds_name:
            label_name = 'gender' 
            label_name = 'all'
            all_y_names = ["age_group", "gender", "glasses"]
            self.meta_df = self.meta_df[self.meta_df["split"] == split].reset_index(drop=True)  

        elif 'CHEXPERT' in self.ds_name:
            label_name = 'Cardiomegaly'
            label_name = 'all'
            all_y_names = [# disease
                            'No Finding', 'Enlarged Cardiomediastinum',
                            'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                            'Pleural Other', 'Fracture', 'Support Devices',
                           # demographic info
                            'age_group_b','sex_b', 'race_b', 'ethnicity_b', 
                           # other info
                            'insurance_type_b','interpreter_needed_b', 
                            'deceased_b','bmi_group_b',
                            ]
            if self.ds_name == 'CHEXPERT_CardioPM':
                all_y_names.append('PM')
                self.meta_df = self.meta_df[self.meta_df["split"] == split].reset_index(drop=True)

            else:
                self.meta_df = self.meta_df[self.meta_df["split_new"] == split].reset_index(drop=True)  

        else:
            raise Exception(f'Not implemented {self.ds_name=}')

        
        self.image_paths = self.meta_df["img_filename"].tolist()

        if label_name == 'all':
            self.labels = []
            for label_col in all_y_names:
                label_values = self.meta_df[label_col].tolist()
                # Map class names to indices and convert to binary (1 and 0)
                unique_classes = sorted(self.meta_df[label_col].unique())
                class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
                binary_labels = [1 if class_to_idx[label] == 1 else 0 for label in label_values]
                self.labels.append(binary_labels)
            self.labels = list(map(list, zip(*self.labels)))
        else:
            self.labels = self.meta_df[label_name].tolist()
            # Map class names to indices
            self.classes = sorted(self.meta_df[label_name].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.labels = [self.class_to_idx[label] for label in self.labels]

    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        img_id = self.image_paths[idx]

        if self.transform:
            image = self.transform(image)
            data = image
        
        if self.encoder_func is not None:
            data = data.unsqueeze(0)
            data = self.encoder_func(data.to(self.device))

        return data, (label, img_id)


class CustomImageFolder(ImageFolder):
    '''
    custommed dataloader, to make sure that the classes index are assigned correctly aligned with NoisyBaseDataModule
    '''
    def __init__(self, root, transform, ds_name=None,encoder_func=None,device=None):
        self.ds_name = ds_name
        super(CustomImageFolder, self).__init__(root, transform)
        self.encoder_func = encoder_func 
        self.device = device
        print(f'{self.ds_name=}')

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Override this method to load from setting file instead of scanning directory
        """
        classes = list(self.classes_to_idx_func().keys())
        classes_to_idx = self.classes_to_idx_func()
        return classes, classes_to_idx
    
    def classes_to_idx_func(self):
        if (self.ds_name == 'CHEXPERT_CardioPM'):
            dict_ = {'non-PM':0,'PM':1}
        elif (self.ds_name == 'MNIST_clean') or (self.ds_name == 'FMNIST_clean'):
            dict_ = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9 }
        elif (self.ds_name == 'cifar10'):
            dict_ = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9 }
        else:
            raise Exception(f'Not implemented {self.ds_name=}')
        return dict_
    
    def __getitem__(self, index: int):
        """
        Override the __getitem__ method to include additional information
        (e.g., file names) in the returned data.
        """
        # Get the default image and label
        path, target = self.samples[index]
        image = self.loader(path)

        if self.transform:
            image = self.transform(image)
            data = image
        
        if self.encoder_func is not None:
            data = data.unsqueeze(0)
            data = self.encoder_func(data.to(self.device))

        extra_info = (target, os.path.basename(path))
        return data, extra_info




    

def get_dataset(dataset_name='MNIST'):

    if len(dataset_name.split('-')) == 1:
        ds_name = dataset_name
        split = 'train'
    else:
        assert len(dataset_name.split('-'))==2
        ds_name, split = dataset_name.split('-')

    if ('MNIST' in dataset_name):
        root = f'{BaseConfig.repo_home_dir}/datasets/{ds_name}/{split}/'
        dataset = CustomImageFolder(root=root,
                                    transform=MNIST_transforms_mnistvar,
                                    ds_name=ds_name)
        
    elif ('cifar10' in dataset_name):
        root = f'{DATASET_DIR}/cifar10/{split}/'
        dataset = CustomImageFolder(root=root,
                                    transform=cifar_transforms,
                                    ds_name=ds_name)

    
        
    elif ('CelebA' in dataset_name):
        data_root = f'{DATASET_DIR}/CelebA/Img/img_align_celeba'
        if 'CelebA_hair_smile' in dataset_name:
            meta_file = f'{DATASET_DIR}/CelebA/Anno/metadata_m_hair_smile.csv'
        else:
            meta_file = f'{DATASET_DIR}/CelebA/Anno/metadata.csv'

        dataset= CustomMetaDataset(ds_name=ds_name,
                                   meta_file=meta_file, 
                                   data_root=data_root, 
                                   transform=celeba_transforms, 
                                   split=split)
    
    elif ('ffhq256' in dataset_name):
        data_root = f'{DATASET_DIR}/ffhq-256/ffhq256/'
        meta_file = f'{DATASET_DIR}/ffhq-256/metadata.csv'
        dataset= CustomMetaDataset(ds_name=ds_name,
                                   meta_file=meta_file, 
                                   data_root=data_root, 
                                   transform=celeba_transforms, 
                                   split=split)
        
    elif ('CHEXPERT' in dataset_name):

        if dataset_name.startswith('OUT'):
            # using the whole chexpert for training
            data_root = CHE_ALL_DATASET_DIR +'preproc_224x224/' 
            meta_file = CHE_ALL_DATASET_DIR + 'metadata_plus_split.csv'

            dataset= CustomMetaDataset(ds_name=ds_name,
                                   meta_file=meta_file, 
                                   data_root=data_root, 
                                   transform=CHE_transforms_che, 
                                   split=split)

        else: 
            # --- CHEXPERT_CardioPM ---

            data_root = f'{CHE_CARDIOPM_DATASET_DIR}{ds_name}/{split}/'
            meta_file = CHE_ALL_DATASET_DIR + 'metadata_plus_split_PM.csv'
            
            dataset= CustomMetaDataset(ds_name=ds_name,
                                   meta_file=meta_file, 
                                   data_root=data_root, 
                                   transform=CHE_transforms_che, 
                                   split=split)
        
    else:
        raise Exception('Not implemented.')
        
    
    return dataset

def get_dataloader(dataset_name='CHEXPERT', 
                   batch_size=32, 
                   pin_memory=False, 
                   shuffle=True, 
                   num_workers=0, 
                   device="cpu"
                  ):
    dataset    = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            pin_memory=pin_memory, 
                            num_workers=num_workers, 
                            shuffle=shuffle
                           )

        
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader

def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0 



#### for getting the dataloader of prototype activation ####
def get_dataset_pact(dataset_name='MNIST',
                     encoder_func = None,
                     device='cpu'):
    if len(dataset_name.split('-')) == 1:
        ds_name = dataset_name
        split = 'train'
    else:
        assert len(dataset_name.split('-'))==2
        ds_name, split = dataset_name.split('-')

    
    if ('CelebA' in dataset_name):
        data_root = f'{DATASET_DIR}/CelebA/Img/img_align_celeba'
        meta_file = f'{DATASET_DIR}/CelebA/Anno/metadata.csv'
        dataset= CustomMetaDataset(ds_name=ds_name,
                                   meta_file=meta_file, 
                                   data_root=data_root, 
                                   transform=celeba_transforms, 
                                   split=split,
                                   encoder_func = encoder_func,
                                   device=device)
        
    elif 'cifar10' in dataset_name:
        root = f'{DATASET_DIR}/cifar10/{split}/'
        dataset = CustomImageFolder(root=root,
                                    transform=cifar_transforms,
                                    ds_name=ds_name,
                                    encoder_func=encoder_func,
                                    device=device,
                                    )
        
    elif 'ffhq256' in dataset_name:
        data_root = f'{DATASET_DIR}/ffhq-256/ffhq256/'
        meta_file = f'{DATASET_DIR}/ffhq-256/metadata.csv'
        dataset= CustomMetaDataset(ds_name=ds_name,
                                   meta_file=meta_file, 
                                   data_root=data_root, 
                                   transform=celeba_transforms, 
                                   split=split,
                                   encoder_func = encoder_func,
                                   device=device)
        
    elif ('MNIST' in dataset_name): # inlcude MNIST and FMNIST
        root = f'{BaseConfig.repo_home_dir}/datasets/{ds_name}/{split}/'
        dataset = CustomImageFolder(root=root,
                                    transform=MNIST_transforms_mnistvar,
                                    ds_name=ds_name,
                                    encoder_func=encoder_func,
                                    device=device,)
        
    elif ('CHEXPERT' in dataset_name):
        if dataset_name.startswith('OUT'):
            # using the whole chexpert for training
            data_root = CHE_ALL_DATASET_DIR +'preproc_224x224/' 
            meta_file = CHE_ALL_DATASET_DIR + 'metadata_plus_split.csv'

            dataset= CustomMetaDataset(ds_name=ds_name,
                                meta_file=meta_file, 
                                data_root=data_root, 
                                transform=CHE_transforms_che, 
                                encoder_func=encoder_func,
                                split=split,
                                device=device)

        else: 
            # --- CHEXPERT_CardioPM ---
            data_root = f'{CHE_CARDIOPM_DATASET_DIR}{ds_name}/{split}/'
            meta_file = CHE_ALL_DATASET_DIR + 'metadata_plus_split_PM.csv'
            
            dataset= CustomMetaDataset(ds_name=ds_name,
                                meta_file=meta_file, 
                                data_root=data_root, 
                                transform=CHE_transforms_che, 
                                encoder_func=encoder_func,
                                split=split,
                                device=device)
        
    else:
        raise Exception(f'Not implemented {dataset_name}')
        
    
    return dataset


def get_dataloader_pact(dataset_name, 
                   batch_size, 
                   pact_encoder, # encoder for prototype
                   pin_memory=False, 
                   shuffle=True, 
                   num_workers=0, 
                   device="cpu"):
    '''
    get the dataloader for the prototype activation
    '''
    dataset    = get_dataset_pact(dataset_name=dataset_name,
                             encoder_func = pact_encoder,
                             device=  device,  
                             )
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            pin_memory=pin_memory, 
                            num_workers=num_workers, 
                            shuffle=shuffle
                           )

        
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader
    


if __name__ == '__main__':
    # test the dataset
    test_ds = 'CelebA'
    print(f'testing on: {test_ds=}')
    print('testing the function: get dataset')
    dataset = get_dataset(dataset_name=test_ds)
    print(f'{len(dataset)=}')


    print('testing the function: get dataloader')
    dataloader = get_dataloader(dataset_name=test_ds, batch_size=32, num_workers=0, device='cpu')
    print(f'{len(dataloader)=}')

    for i, (img, label) in enumerate(dataloader):
        print(f'{img.shape=}')
        print(f'{label=}')
        break


    

