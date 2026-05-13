import glob
import itertools
import logging
import math
import os
import shutil
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.nn import knn
from torch_scatter import scatter_mean

from pbc_config import BOX, min_image

### EMBEDDING UTILITIES ###
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, t_embd_dim):
        super().__init__()
        self.t_embd_dim = t_embd_dim

    def forward(self, t):
        # t: (batch_size, 1) — timestep values
        t = t * 850
        half = self.t_embd_dim // 2
        freqs = torch.exp(
            -math.log(10) * torch.arange(half, device=t.device) / (half - 1) # originally 10000
        )                                          # (half,)
        args = t * freqs                           # (batch_size, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (batch_size, t_embd_dim)
        return emb
    
class SinusoidalThetaEmbedding(nn.Module):
    def __init__(self, theta_param_embd_dim):
        super().__init__()
        self.theta_param_embd_dim = theta_param_embd_dim

    def forward(self, theta):
        # t: (batch_size, 1) — timestep values
        N, D = theta.shape
        half = self.theta_param_embd_dim // 2
        freqs = torch.exp(
            -math.log(2) * torch.arange(half, device=theta.device) / (half - 1)
        )                                          # (half,)
        embeddings = []
        for param in range(D):
            args = theta[:, param].unsqueeze(1) * freqs        # (batch_size, half)
            emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (batch_size, theta_param_embd_dim)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=1)

### DATA UTILITIES ###
def compute_local_density(edge_index, x1):
    row, col = edge_index
    coord_diff = min_image(x1[row] - x1[col], **BOX)
    distances = torch.norm(coord_diff, dim=1, keepdim=True)
    mean_dist = scatter_mean(distances, row, dim=0, dim_size=x1.shape[0])
    # print(f"Mean distance shape: {mean_dist.shape}")
    return mean_dist

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def radius_graph_pbc_batch(x, r, batch, box_size=1.0):
    edge_indices = []
    for i in range(batch.max().item() + 1):
        mask = batch == i
        xi = x[mask]
        
        diff = xi.unsqueeze(0) - xi.unsqueeze(1)
        diff = min_image(diff, **BOX)
        dist = diff.norm(dim=-1)
        
        edges = ((dist < r) & (dist > 0)).nonzero().t()
        
        offset = mask.nonzero()[0].item()
        edges = edges + offset
        
        edge_indices.append(edges)
    
    return torch.cat(edge_indices, dim=1)

def gpu_knn_graph_pbc_batch(x, k, batch_y, device):
    """KNN periodic boundary conditions for a single batch (graph)"""

    # Generate all 27 periodic shifts
    shifts = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), device=device)
    x_extended = (x.unsqueeze(1) + shifts.unsqueeze(0)).view(-1, 3)  # (27*N, 3) view flattens it
    batch_x = batch_y.repeat_interleave(shifts.shape[0]) # extended each graph batch by * 27

    # Query k+1 neighbors to account for self-match exclusion
    knn_indices = knn(x_extended, x, k+1, batch_x, batch_y)

    # Separate query and neighbor indices
    query_indices = knn_indices[0] 
    neighbor_indices = knn_indices[1]  

    # Convert to original indices
    num_shifts = shifts.shape[0]  # 27
    original_indices = neighbor_indices // num_shifts  # Map back to original indices

    # Remove self-matches
    mask = query_indices != original_indices
    query_indices, original_indices = query_indices[mask], original_indices[mask]

    return torch.stack([query_indices, original_indices], dim=0)

def gpu_knn_graph_batch(x, k, batch, device):
    """KNN periodic boundary conditions for a single batch (graph)"""
    
    knn_indices = knn(x, x, k + 1, batch, batch)

    query_indices = knn_indices[0]
    neighbor_indices = knn_indices[1]

    mask = query_indices != neighbor_indices
    query_indices = query_indices[mask]
    neighbor_indices = neighbor_indices[mask]

    return torch.stack([query_indices, neighbor_indices], dim=0)

def find_equally_spaced_indices(arr, n):
    "Based on specific conditioning parameter."
    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]
    min_val, max_val = sorted_arr[0], sorted_arr[-1]
    division_points = np.linspace(min_val, max_val, n)
    closest_indices = np.searchsorted(sorted_arr, division_points)
    closest_indices = np.clip(closest_indices, 0, len(arr) - 1)
    original_indices = sorted_indices[closest_indices]
    return original_indices

def scale_thetas(theta):
    mins = torch.tensor([0.1003, 0.03001, 0.5001, 0.8001, 0.6001]).to(theta.device)
    maxs = torch.tensor([0.4999, 0.06999, 0.8999, 1.1999, 0.9995]).to(theta.device)
    return 2 * (theta - mins) / (maxs - mins) - 1

def scale_masses(masses):
    min_mass = 13.6390
    max_mass = 16.1808
    scaled_masses = (masses - min_mass) / (max_mass - min_mass)
    return scaled_masses

def unscale_masses(scaled_masses):
    min_mass = 13.6390
    max_mass = 16.1808
    masses = scaled_masses * (max_mass - min_mass) + min_mass
    return masses

def ot_alignment(x0, x1): 
    x1_ot = x1.clone()
    for b in range(x0.shape[0]):
        diff = x0[b].unsqueeze(1) - x1[b].unsqueeze(0)  # (N, N, 3)
        diff = min_image(diff, **BOX)
        cost = (diff ** 2).sum(-1)  # (N, N)
        _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        x1_ot[b] = x1[b, col_ind]
    
    return x1_ot

def ot_alignment_variable(x0, x1, batch, batch_size):
    aligned_x0 = []
    for i in range(batch_size):
        mask = batch == i
        g0 = x0[mask]
        g1 = x1[mask]
        
        diff = g1.unsqueeze(1) - g0.unsqueeze(0)
        diff = min_image(diff, **BOX)
        cost = (diff ** 2).sum(-1)
        _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        
        aligned_x0.append(g0[col_ind])
    
    return torch.cat(aligned_x0, dim=0)

### LOGGING UTILITIES ###
def setup_logging(config, output_dir):
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

def setup_logging_infer(config, output_dir):
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, "infer.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

def unique_output_dir(config):
    # Generate a unique directory name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output"]["base_path"], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def unique_output_dir_infer(output_dir, config):
    # Generate a unique directory name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f'{config["inference"]["infer_type"]}/{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def copy_config_to_output(config_path, output_dir):
    """
    Copies the configuration file to the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_path, output_dir)

### CONFIG LOADING UTILITIES ###
def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def merge_configs(yaml_config, cli_args):
    """Merge YAML config with CLI arguments (CLI takes priority)."""
    config = yaml_config.copy()
    
    for key, value in vars(cli_args).items():
        if value is not None:  # Only override if CLI value is provided
            config[key] = value

    return config

### TRAINING UTILITIES ###
def gradient_norm(model):

    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:  # Ensure the parameter has gradients
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm ** 0.5
    return total_norm 

def adjust_learning_rate(optimizer, epoch, config):
    """Adjusts learning rate for each epoch during the warm-up phase, based on config."""
    warmup_epochs = config["training"]["scheduler"]["warmup_epochs"]
    if epoch < warmup_epochs:
        initial_lr = float(config["training"]["scheduler"]["initial_lr"])
        target_lr = float(config["training"]["scheduler"]["target_lr"])
        current_lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

def get_scheduler(optimizer, config):
    type = config["training"]["scheduler"]["type"]
    
    if type == "plateau":
        decay_scheduler = ReduceLROnPlateau(optimizer, 
                                            factor=config["training"]["scheduler"]["factor"],
                                            threshold=config["training"]["scheduler"]["threshold"], 
                                            min_lr=config["training"]["scheduler"]["min_lr"], 
                                            patience=config["training"]["scheduler"]["patience"])
    elif type == "cosine":
        decay_scheduler = CosineAnnealingLR(optimizer, 
                                            T_max=config["training"]["epochs"] + 1 - config["training"]["scheduler"]["warmup_epochs"],
                                            eta_min=1e-7)
    else:
        raise ValueError(f"Unknown scheduler: {type}")
    
    return decay_scheduler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_activation_fn(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")
    
def get_output_activation_fn(name):
    # Pred scores should be in range [-1, 1]
    if name == "negative_allowed":
        return torch.nn.Tanh()
    # Pred scores should be positive, the targets will be in range [0, 1]
    elif name == "positive_only":
        return torch.nn.Softplus()
    # Do not normalize scores
    elif name is None:
        return None
    else:
        raise NameError(f"Unknown value for norm_score: {name}")
    
def get_aggregation_fn(name):
    if name == "sum":
        return unsorted_segment_sum
    elif name == "mean":
        return unsorted_segment_mean
    else:
        raise ValueError(f"Unknown aggregation function: {name}")
    
def init_weights(m, method="xavier_uniform", mode="fan_in", gain=5):
    if isinstance(m, nn.Linear):
        if method == "xavier_uniform":
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        elif method == "xavier_normal":
            torch.nn.init.xavier_normal_(m.weight)
        elif method == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(m.weight, mode=mode, a=-0.5)
            print(f"Initialization method with {method} and mode fan out")
        elif method == "kaiming_normal":
            torch.nn.init.kaiming_normal_(m.weight, mode=mode)
        elif method == "orthogonal":
            torch.nn.init.orthogonal_(m.weight)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -0.1, 0.1)
            # m.bias.data.fill_(0.01)

def load_yaml_from_directory(directory):
    # Find all YAML files in the directory
    yaml_files = glob.glob(os.path.join(directory, "*.yaml")) + glob.glob(os.path.join(directory, "*.yml"))
    
    if not yaml_files:
        raise FileNotFoundError(f"No YAML file found in {directory}")
    
    # Load the first found YAML file
    yaml_path = yaml_files[0]  # Assuming there is only one YAML file
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    
    return data