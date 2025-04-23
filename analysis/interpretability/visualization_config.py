
from dataclasses import dataclass

@dataclass
class VisConfig:
    '''
    Which trained model you visualize prototype on.
    '''
    ds_name: str = 'CelebA' # <-- set here
    version_num: int =  4 # <-- set here
    num_random_pick_samples: int = 20 # randomly select n samples, and get the most activated patch from all, by default 20
    vis_p_ind: tuple[int] = (10,11,12,13,14,15,16,17,18,19) # collection of p_ind for visulization
    seed: int = None # add seed to fix the random selection in different run

    
