import numpy as np
import torch
import time
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment
from pbc_config import min_image, BOX

def periodic_cost_squared(x0, x1, box_size=1.0):
    diff = x0[:, None, :] - x1[None, :, :]
    diff = min_image(diff, **BOX)
    return (diff ** 2).sum(-1)

halos = np.load("/gpfs/home4/bartb/T5000/Data/train_halos.npy")
N = len(halos)
N_halos = halos.shape[1]
x0_paired = np.zeros((N, N_halos, 3), dtype=np.float32)

# Uniform weights for OT
a = np.ones(N_halos, dtype=np.float64) / N_halos
b = np.ones(N_halos, dtype=np.float64) / N_halos

# Time first sample
x1 = torch.tensor(halos[0][:, :3] / 1000., dtype=torch.float32)
x0 = torch.rand_like(x1)
# t0 = time.time()
C = periodic_cost_squared(x0, x1).numpy().astype(np.float64)
_, pi = linear_sum_assignment(C)
inv_pi = np.argsort(pi)
# t_first = time.time() - t0
# print(f"First sample: {t_first:.2f}s | Estimated total: {t_first * N / 3600:.1f}h")
x0_paired[0] = x0[inv_pi].numpy()

for idx in tqdm(range(1, N)):
    x1 = torch.tensor(halos[idx][:, :3] / 1000., dtype=torch.float32)
    x0 = torch.rand_like(x1)
    C = periodic_cost_squared(x0, x1).numpy().astype(np.float64)
    _, pi = linear_sum_assignment(C)
    inv_pi = np.argsort(pi)
    x0_paired[idx] = x0[inv_pi].numpy()

    if idx % 99 == 0:
        print(f"Processed {idx+1}/{N} samples.")

np.save("/gpfs/home4/bartb/T5000/Data/train_x0_ot.npy", x0_paired)