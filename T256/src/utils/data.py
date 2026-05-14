import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pbc_config import BOX, min_image

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

def ot_alignment(x0, x1, batch_size): 
    n_halos = x0.shape[0] // batch_size
    dim = x0.shape[1]

    x0 = x0.view(batch_size, n_halos, dim)
    x1 = x1.view(batch_size, n_halos, dim)

    aligned_x0 = []
    for b in range(batch_size):
        g0 = x0[b]
        g1 = x1[b]

        diff = g1.unsqueeze(1) - g0.unsqueeze(0)
        diff = min_image(diff, **BOX)
        cost = (diff ** 2).sum(-1)
        _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        aligned_x0.append(g0[col_ind])

    return torch.cat(aligned_x0, dim=0)

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