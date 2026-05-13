import glob
import os
import yaml

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

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