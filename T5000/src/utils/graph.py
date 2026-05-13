import itertools

import torch
from torch_geometric.nn import knn
from torch_scatter import scatter_mean

from pbc_config import BOX, min_image

def compute_local_density(edge_index, x1):
    row, col = edge_index
    coord_diff = min_image(x1[row] - x1[col], **BOX)
    distances = torch.norm(coord_diff, dim=1, keepdim=True)
    mean_dist = scatter_mean(distances, row, dim=0, dim_size=x1.shape[0])
    return mean_dist

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