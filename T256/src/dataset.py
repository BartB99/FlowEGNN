from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from pbc_config import BOX, wrap
from utils.scale import scale_masses

def define_rotations():
    signs = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ], dtype=np.float32)

    perms = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ], dtype=np.int64)

    sign = signs[np.random.randint(0, 8)]
    perm = perms[np.random.randint(0, 6)]

    matrix = np.eye(3)[perm] * sign
    return matrix


def augment_data(x, rotations, translations):
    if rotations:
        matrix = define_rotations()
        x = np.dot(x, matrix.T)

    if translations:
        shift = np.random.uniform(size=(3,)) # sample a random shift for each dimension
        x = x + shift
    
    return wrap(x, **BOX) # ensure pbc

def augment_data_paired(x, x0, rotations, translations):
    if rotations:
        matrix = define_rotations()
        x = np.dot(x, matrix.T)
        x0 = np.dot(x0, matrix.T)

    if translations:
        shift = np.random.uniform(size=(3,))
        x = x + shift
        x0 = x0 + shift
    
    x = wrap(x, **BOX)
    x0 = wrap(x0, **BOX)
    
    return x, x0

class CosmologyData(Data):
    """
    Batch along new dimension for graph-level features.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Whatever values "theta" and "z" hold, they are graph-level features
        if key == 'theta':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
    

class T256Dataset(Dataset):

    def __init__(self, cosmologies_dir, cosmologies_info_dir, split, 
                    rotations=False, translations=False, cosm_param=None):
        """
        split (str): "train" or "test"
        """
        super().__init__()
        
        self.halos = np.load(Path(cosmologies_dir) / f"{split}_subbox_halos.npy")
        self.subbox_counts = np.load(Path(cosmologies_dir) / f"{split}_subbox_counts.npy")

        # Load cosmlogical parameters
        params = pd.read_csv(Path(cosmologies_info_dir) / f"{split}_cosmology.csv")
        if cosm_param is None:
            self.params = params.values
        else:
            self.params = params[cosm_param].values
        
        self.rotations = rotations
        self.translations = translations

    def len(self):
        return len(self.params)

    def get(self, idx):

        # Load the dataset file for the specific index
        graph = self.halos[idx]
        n = self.subbox_counts[idx]
        params = self.params[idx]

        data_point = graph[:n]

        mass = torch.tensor(data_point[:, -1], dtype=torch.float32)
        log_mass = torch.log10(mass)
        scaled_log_mass = scale_masses(log_mass)

        x = (data_point[:, :3] - 0.) / (370. - 0.)
        
        x = augment_data(x, rotations=self.rotations, translations=self.translations)

        graph = CosmologyData(
            mass=scaled_log_mass,  # mass
            x=torch.tensor(x, dtype=torch.float32),  # position
            vel=torch.tensor(data_point[:, 3:6], dtype=torch.float32),  # velocity
            theta=torch.tensor(params, dtype=torch.float32),  # cosmological parameters
        )
            
        return graph
    
def deterministic_sample(cosmologies_dir="Data", cosmologies_info_dir="Data",
                         rotations=False, translations=False,
                         cosm_param=False):
    valid_dataset = T256Dataset(cosmologies_dir, cosmologies_info_dir, 
                                 split="test", rotations=rotations, 
                                 translations=translations, cosm_param=None)

    return valid_dataset[0]  # Return the first sample for deterministic testing

def create_dataloader(cosmologies_dir, 
                      cosmologies_info_dir, 
                      distributed,
                      train_kwargs=None,
                      valid_kwargs=None,
                      rotations=False, 
                      translations=False,
                      cosm_param=None):

    train_dataset = T256Dataset(cosmologies_dir, cosmologies_info_dir, split="train", translations=translations, cosm_param=cosm_param)
    valid_dataset = T256Dataset(cosmologies_dir, cosmologies_info_dir, split="test", rotations=rotations, translations=translations, cosm_param=cosm_param)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, drop_last=True, **train_kwargs)
    else:
        train_sampler = None
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **train_kwargs) 
    
    valid_loader = DataLoader(valid_dataset, shuffle=True, drop_last=True, **valid_kwargs)

    return train_loader, valid_loader, train_sampler    