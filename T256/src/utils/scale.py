import torch

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