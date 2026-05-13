import logging
import math
import os
import pprint
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import yaml
from torch.amp import autocast

from egnn import EGNN
from fm import FlowMatching
from mlp_baseline import MLPBaseline
from pbc_config import BOX, wrap
from prior import scaled_log10_gaussian_mass_prior, uniform_prior, wrapped_gaussian_prior
from utils.config import load_config
from utils.data import find_equally_spaced_indices
from utils.graph import compute_local_density, radius_graph_pbc_batch
from utils.logging import setup_logging_infer, unique_output_dir_infer
from utils.scale import scale_thetas
from utils.training import get_activation_fn, load_yaml_from_directory

def parse_arguments():
    """Parses CLI arguments."""
    parser = ArgumentParser(description="Training script with YAML and CLI support.")

    # YAML Configuration File (Mandatory)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

    return parser.parse_args()

def initialize_inference(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    input_theta_d = len(config["data"]["cosm_param"]) if config["data"]["cosm_param"] else 5

    egnn = EGNN(t_embed_dim=config["model"]["fm"]["t_embed_dim"],
                input_node_d=config["model"]["egnn"]["input_node_d"],
                input_theta_d=input_theta_d,
                theta_param_embd_dim=config["model"]["egnn"]["theta_param_embd_dim"],
                hidden_nf=config["model"]["egnn"]["hidden_nf"],
                latent_nf=config["model"]["egnn"]["latent_nf"],
                theta_nf=config["model"]["egnn"]["theta_nf"],
                n_layers=config["model"]["egnn"]["n_layers"],
                mlp_layers=config["model"]["egnn"]["mlp_layers"],
                single_layer=config["model"]["egnn"]["single_layer"],
                recurrent=config["model"]["egnn"]["recurrent"],
                activation=get_activation_fn(config["model"]["egnn"]["activation"]),
                norm=config["model"]["egnn"]["norm"],
                attention=config["model"]["egnn"]["attention"],
                scale_pred=config["model"]["egnn"]["scale_pred"],
                coords_weight=config["model"]["egnn"]["coords_weight"],
                norm_diff=config["model"]["egnn"]["norm_diff"])
    
    mlp = MLPBaseline(
                t_embed_dim=config["model"]["fm"]["t_embed_dim"],
                input_node_d=config["model"]["egnn"]["input_node_d"],
                input_theta_d=input_theta_d,
                theta_param_embd_dim=config["model"]["egnn"]["theta_param_embd_dim"],
                hidden_nf=1024,
                n_mlp_layers=2,
            )
    
    model = FlowMatching(sigma_0=config["model"]["fm"]["sigma_0"],
                         sigma_sched=config["model"]["fm"]["sigma_sched"],
                         t_embed_dim=config["model"]["fm"]["t_embed_dim"],
                         version=config["model"]["fm"]["version"],
                         vnet=egnn,
                         batch_size=config["training"]["batch_size"],
                         prior=config["model"]["fm"]["prior"],
                         k=config["model"]["fm"]["k"],
                         r=config["model"]["fm"]["r"],
                         optimal_transport=config["model"]["fm"]["optimal_transport"],
                         lambda_mass=config["model"]["fm"]["lambda_mass"],
                         t_embed=config["model"]["fm"]["t_embed"],
                         dim=3
                         ).to(device)  
        
    checkpoint = torch.load(config["inference"]["checkpoint_path"], map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items()}  # Remove "module."
    model.load_state_dict(new_state_dict)

    print("Loaded model")

    return device, model


def sample(x0, batch, conditioning, model, r, device, config):
    ts = torch.linspace(0, 1, config["inference"]["T"], device=device)
    dt = ts[1] - ts[0]

    # h_in = torch.ones(x0.shape[0], 1, device=device)
    # h_in = torch.normal(mean=0.3499, std=0.1008, size=(x0.shape[0], 1), device=device)  # Sample masses using the mean and std of the training data masses (after log10 and scaling)

    m0 = scaled_log10_gaussian_mass_prior(torch.empty(x0.shape[0], 1, device=device))  
    m_t = m0
    x_t = x0
    
    model.eval()
    with torch.no_grad():
        for step, t in enumerate(ts):
            edge_index = radius_graph_pbc_batch(x_t, r, batch, device)
            density = compute_local_density(edge_index, x_t)

            h_in = torch.cat([m_t, density], dim=1)  # Concatenate local density and mass as node features

            t = t.view(1, 1)

            num_graphs = int(batch.max().item()) + 1
            t_embd = model.time_embedding(t).expand(num_graphs, -1)

            with autocast(device_type="cuda", dtype=torch.float16):

                pred_v, pred_v_m = model.vnet(h=h_in,
                                              x=x_t, 
                                              t_embed=t_embd,
                                              batch=batch,
                                              edge_index=edge_index,
                                              theta=conditioning)

            x_t = x_t + pred_v * dt
            x_t = wrap(x_t, **BOX)

            m_t = m_t + pred_v_m * dt
    
    x_final = x_t
    m_final = m_t

    return x_final, m_final


def which_cosmologies(config):
    """Determines which for which values of the cosmological parameters to sample."""

    # load cosmological param/ conditioning
    conditioning = pd.read_csv(f'{config["data"]["cosmologies_info_dir"]}test_cosmology.csv').values


    if config["inference"]["infer_type"] == "qualitative_2pcf":
        
        equally_spaced = 20
        ### equally spaced based on Omega_m! : conditioning[:, 0]
        conditioning_idx_used = find_equally_spaced_indices(conditioning[:,0], equally_spaced)
        conditioning = conditioning[conditioning_idx_used]

        conditioning = conditioning[::4] #[:, [0, -1]]

    elif config["inference"]["infer_type"] == "quantitative_2pcf":
        
        if config["data"]["cosm_param"] is not None:
            conditioning = conditioning[:, [0, -1]]

    else:
        raise NameError("Unknown type or purpose of inference")

    return conditioning


def infer(config, output_dir):

    device, model = initialize_inference(config)

    conditioning_used = which_cosmologies(config)

    n_cosmologies = conditioning_used.shape[0]  

    n_repeats = config["inference"]["n_repeats"]

    prior = config["model"]["fm"]["prior"]

    n_halos = config["inference"]["inference_halo_count"]

    batch_size = config["inference"]["batch_size"]
    total_n_cosmologies = n_repeats * n_cosmologies
    n_batches = math.ceil(total_n_cosmologies / batch_size)

    generated_samples = []
    generated_masses = []

    for i in range(n_batches):

        if config["inference"]["infer_type"] == "quantitative_2pcf":

            print(conditioning_used.shape)

            # if conditioning_used.shape[0] == 200:
            n_cosmologies = batch_size // n_repeats
            lower_bound = i * n_cosmologies
            upper_bound = (i+1) * n_cosmologies
            # else:
            #     lower_bound = 0
            #     upper_bound = conditioning_used.shape[0]

            if prior == "uniform":
                x0 = uniform_prior(torch.empty(n_halos * n_repeats * n_cosmologies, 3, dtype=torch.float32, device=device))
            elif prior == "gaussian":
                x0 = wrapped_gaussian_prior(torch.empty(n_halos * n_repeats * n_cosmologies, 3, dtype=torch.float32, device=device))
            else:
                raise ValueError(f"Unknown prior: {prior}")
            
            batch = torch.arange(n_repeats * n_cosmologies, device=device).repeat_interleave(n_halos)
            cond_infer = torch.tensor(conditioning_used[lower_bound: upper_bound], dtype=torch.float32, device=device).repeat_interleave(n_repeats, dim=0)
            cond_infer = scale_thetas(cond_infer)

            gen_samples, gen_masses = sample(x0=x0, batch=batch, conditioning=cond_infer, model=model, r=config["model"]["fm"]["r"], device=device, config=config)
            
            generated_samples.append(gen_samples.reshape(n_cosmologies, n_repeats, n_halos, 3).detach().cpu().numpy())
            generated_masses.append(gen_masses.reshape(n_cosmologies, n_repeats, n_halos, 1).detach().cpu().numpy())
            print(f"Generated samples shape: {gen_samples.shape}")

        elif config["inference"]["infer_type"] == "qualitative_2pcf":
            mean_halo_counts = [281, 252, 235, 264, 250]

            for i in range(n_cosmologies):
                n_halos_i = mean_halo_counts[i]

                if prior == "uniform":
                    x0 = uniform_prior(torch.empty(n_halos_i * n_repeats, 3, dtype=torch.float32, device=device))
                elif prior == "gaussian":
                    x0 = wrapped_gaussian_prior(torch.empty(n_halos_i * n_repeats, 3, dtype=torch.float32, device=device))
                else:
                    raise ValueError(f"Unknown prior: {prior}")
                
                batch = torch.arange(n_repeats, device=device).repeat_interleave(n_halos_i)
                cond_infer = torch.tensor(conditioning_used[i:i+1], dtype=torch.float32, device=device).repeat_interleave(n_repeats, dim=0)
                cond_infer = scale_thetas(cond_infer)

                gen_samples, gen_masses = sample(x0=x0, batch=batch, conditioning=cond_infer, model=model, r=config["model"]["fm"]["r"], device=device, config=config)
                
                generated_samples.append(gen_samples.reshape(1, n_repeats, n_halos_i, 3).detach().cpu().numpy())
                generated_masses.append(gen_masses.reshape(1, n_repeats, n_halos_i, 1).detach().cpu().numpy())
                print(f"Generated samples shape: {gen_samples.shape}")

            break

        else:
            raise NameError("Unknown type or purpose of inference")

    print(f"Finished")
    
    # final_samples = np.array(generated_samples)
    # final_masses = np.array(generated_masses)   

    torch.save(generated_samples, f'{output_dir}/gen_samples.pth')
    torch.save(generated_masses, f'{output_dir}/gen_masses.pth')
    torch.save(conditioning_used, f'{output_dir}/cond.pth')

    return gen_samples


if __name__ == "__main__":
    args = parse_arguments()
    
    infer_config = load_config(args.config)
    output_dir = os.path.dirname(infer_config["inference"]["checkpoint_path"])
    train_config = load_yaml_from_directory(output_dir)

    final_config = {**infer_config, **train_config}  # The second file's keys overwrite the first file's keys if they overlap
    
    output_dir = unique_output_dir_infer(output_dir, final_config) 

    # Define the path where you want to save the final config
    final_config_path = os.path.join(output_dir, "final_config.yaml")

    # Save the final configuration as a YAML file
    with open(final_config_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False)


    setup_logging_infer(final_config, output_dir)
    logging.info(f"Output_dir: {output_dir}")
    logging.info("Final training configuration:\n%s", pprint.pformat(final_config))

    logging.info(f'Inference on run: {final_config["inference"]["checkpoint_path"]}')
    logging.info(f'Inference on {final_config["model"]["fm"]["version"]} model')
    logging.info(f'Inference for {final_config["inference"]["infer_type"]} evaluation')
    infer(final_config, output_dir)