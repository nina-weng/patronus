import torch
from tqdm  import tqdm
import matplotlib.pyplot as plt
from train.dataloader import inverse_transform
import numpy as np

def get_samples_from_loader(loader, selected_sample_id):
    """
    Get samples from a dataloader.
    """
    num_selected_sample = len(selected_sample_id)
    # Create a mapping of IDs to indices for easy lookup
    id_to_index = {img_id: idx for idx, img_id in enumerate(selected_sample_id)}

    # Initialize an empty tensor for selected images (assuming all images have the same shape)
    example_img_shape = next(iter(loader))[0][0].shape  # Get the shape of a single image
    selected_img = torch.zeros((num_selected_sample, *example_img_shape))  # Empty tensor to store selected images

    # Track which IDs have been matched
    found_ids = set()

    # Loop through the test dataloader
    for b_image, extra_info in tqdm(loader):
        b_img_id = extra_info[1]  # Assuming this contains the image IDs
        # Find indices of matching IDs in the batch
        matching_indices = [i for i, img_id in enumerate(b_img_id) if img_id in id_to_index]
        
        if matching_indices:  # Only process if there are matches
            for i in matching_indices:
                idx = id_to_index[b_img_id[i]]  # Find the correct index in `selected_img`
                if b_img_id[i] not in found_ids:  # Check if this ID is already processed
                    selected_img[idx] = b_image[i]  # Place the image in the correct position
                    found_ids.add(b_img_id[i])  # Mark the ID as found
            print(f'Found selected samples: {[b_img_id[i] for i in matching_indices]}')
        
        if len(found_ids) == num_selected_sample:  # Stop only when all IDs are found
            break

    return selected_img


def vis_samples(selected_img, selected_sample_id):
    num_selected_sample = len(selected_img)

    def process_image(img):
        img = inverse_transform(img).type(torch.uint8).cpu().squeeze()
        img_np = img.numpy()
        # If image is 2D, it's grayscale; otherwise, it's (C, H, W)
        if img_np.ndim == 2:
            return img_np  # grayscale
        elif img_np.shape[0] == 1:
            return img_np[0]  # still grayscale, (1, H, W) -> (H, W)
        else:
            return np.transpose(img_np, (1, 2, 0))  # RGB or similar

    if num_selected_sample > 1:
        fig, axes = plt.subplots(1, num_selected_sample, figsize=(num_selected_sample * 4, 4))
        for i in range(num_selected_sample):
            img_to_show = process_image(selected_img[i])
            axes[i].imshow(img_to_show, cmap='gray' if img_to_show.ndim == 2 else None)
            axes[i].axis('off')
            axes[i].set_title(f'{selected_sample_id[i]}')
        plt.show()
    elif num_selected_sample == 1:
        img_to_show = process_image(selected_img[0])
        plt.imshow(img_to_show, cmap='gray' if img_to_show.ndim == 2 else None)
        plt.axis('off')
        plt.title(f'{selected_sample_id[0]}')
        plt.show()