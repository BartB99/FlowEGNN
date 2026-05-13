import logging
import os
import pprint
import sys
from argparse import ArgumentParser
from socket import gethostname

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from dataset import create_dataloader
from egnn import EGNN
from fm import FlowMatching
from mlp_baseline import MLPBaseline
from utils.config import load_config, merge_configs
from utils.logging import copy_config_to_output, setup_logging, unique_output_dir
from utils.training import adjust_learning_rate, count_parameters, get_activation_fn, get_scheduler, gradient_norm

### Distributed Training Functions ####################################
def setup(rank, world_size):
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
#######################################################################
    
def parse_arguments():
    """Parses CLI arguments."""
    parser = ArgumentParser(description="Training script with YAML and CLI support.")

    # YAML Configuration File (Mandatory)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

    # Optional CLI Overrides
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, help="Number of training epochs")
    parser.add_argument('-lr', '--learning_rate', type=float, help="Learning rate")
    parser.add_argument('--clip_value', type=float, help="Gradient clipping value")
    parser.add_argument('--log_wandb', action='store_true', help="Whether to log the run")

    return parser.parse_args()

def initialize_training(config, rank, local_rank):
    """Initializes model, optimizer, and training parameters."""
    
    ######################## DDP ##############################
    train_kwargs = {'batch_size': config["training"]["batch_size"]}
    test_kwargs = {'batch_size': config["training"]["batch_size_valid"]}

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    num_workers = int(slurm_cpus) if slurm_cpus is not None else max(os.cpu_count() or 1, 1)
    cuda_kwargs = {'num_workers': num_workers,
                    'pin_memory': True}
    
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    wandb.init(project=config["wandb"]["project_name"], 
               name=config["wandb"]["run_name"],
               config=config) if config["log_wandb"] and rank == 0 else None

    # Define data
    train_dl, valid_dl, train_sampler = create_dataloader(
        cosmologies_dir=config["data"]["cosmologies_dir"],
        cosmologies_info_dir=config["data"]["cosmologies_info_dir"],
        distributed=True,    ########## DDP ###########
        train_kwargs=train_kwargs, 
        valid_kwargs=test_kwargs,
        rotations=config["data"]["rotations"],
        translations=config["data"]["translations"],
        cosm_param=config["data"]["cosm_param"],
    )

    input_theta_d = len(config["data"]["cosm_param"]) if config["data"]["cosm_param"] is not None else 5

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
                n_mlp_layers=4,
            )

    model = FlowMatching(sigma_0=config["model"]["fm"]["sigma_0"],
                         sigma_sched=config["model"]["fm"]["sigma_sched"],
                         t_embed_dim=config["model"]["fm"]["t_embed_dim"],
                         version=config["model"]["fm"]["version"],
                         vnet=egnn,
                         batch_size=config["training"]["batch_size"],
                         prior=config["model"]["fm"]["prior"],
                         k = config["model"]["fm"]["k"],
                         r = config["model"]["fm"]["r"],
                         optimal_transport = config["model"]["fm"]["optimal_transport"],
                         lambda_mass = config["model"]["fm"]["lambda_mass"],
                         t_embed=config["model"]["fm"]["t_embed"],
                         dim=3
                    ).to(local_rank)  ########### DDP #############
    

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) ########### DDP #############
    if rank == 0:
        logging.info(f"Number of trainable parameters: {count_parameters(ddp_model)}") 
        logging.info("Model architecture: %s", ddp_model)

    # Define optimization
    optimizer = AdamW(ddp_model.parameters(), ########### DDP #############
                      lr=float(config["training"]["scheduler"]["initial_lr"]), 
                       weight_decay=config["training"]["optimizer"]["weight_decay"])
    scheduler_config = config["training"]["scheduler"]
    decay_scheduler = get_scheduler(optimizer, config) \
          if scheduler_config["type"] is not None else None

    # check whether to load from checkpoint
    if not config["training"]["start_from_scratch"]:
        if (
            "checkpoint_path" not in config["training"]
            or not config["training"]["checkpoint_path"]
        ):
            if rank == 0:
                logging.error(
                    "Checkpoint path must be provided when resuming from a checkpoint."
                )
                sys.exit(
                    "Error: Checkpoint path not provided but required for resuming training."
                )
        elif not os.path.exists(config["training"]["checkpoint_path"]):
            if rank == 0:
                logging.error(
                    f"Checkpoint file not found: {config['training']['checkpoint_path']}"
                )
                sys.exit("Error: Checkpoint file does not exist.")
        else:
            map_location = f"cuda:{local_rank}"  # Ensure each rank loads onto its own GPU
            checkpoint = torch.load(config["training"]["checkpoint_path"], map_location=map_location)
            ddp_model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            decay_scheduler.load_state_dict(checkpoint["decay_scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            if rank == 0:
                logging.info("Resuming training from checkpoint.")
    else:
        start_epoch = 1
        if (
            "checkpoint_path" in config["training"]
            and config["training"]["checkpoint_path"]
        ):
            if rank == 0:
                logging.warning(
                    "Checkpoint path provided but will not be used since training starts from scratch."
                )
    
    return train_dl, valid_dl, train_sampler, ddp_model, optimizer, decay_scheduler, start_epoch


def evaluate_early_stopping(running_loss, best_loss, patience_counter, min_delta):
    """Checks early stopping conditions and returns updated values."""
    if running_loss < best_loss - min_delta:
        return running_loss, 0
    else:
        patience_counter += 1
        return best_loss, patience_counter


def save_model(model, model_name, optimizer, decay_scheduler, epoch, output_dir):
    """Saves the model checkpoint locally and return filepath."""
    file_path = os.path.join(output_dir, model_name)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "decay_scheduler_state": decay_scheduler.state_dict() \
              if decay_scheduler is not None else None,
        "epoch": epoch,
    }
    torch.save(checkpoint, file_path)
    return file_path


def train_epoch(ddp_model, train_dl, optimizer, device, config, epoch, rank, global_step):
    """Trains the model for one epoch and returns the average training loss for the epoch."""
    
    ddp_model.train()
    train_loss = 0.0
    train_pos_loss = 0.0
    train_mass_loss = 0.0

    for i, samples in enumerate(train_dl):
        
        optimizer.zero_grad()
        samples = samples.to(device)
        loss, pos_loss, mass_loss = ddp_model(samples=samples)  

        train_loss += loss.item()
        train_pos_loss += pos_loss
        train_mass_loss += mass_loss

        loss.backward()

        # for name, param in ddp_model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad_norm={param.grad.norm().item():.2e}")
        #     else:
        #         print(f"{name}: grad=None")

        total_norm = gradient_norm(ddp_model)
        
        global_step += 1  # Increase step at each batch
        wandb.log({"gradient_norm": total_norm, "train/batch_loss": loss.item()}, \
                  step=global_step) if config["log_wandb"] and rank == 0 else None
        if config["log_wandb"] and rank == 0:
            wandb.log({
                "gradient_norm": total_norm, 
                "train/batch_loss": loss.item(),
                "train/pos_loss": pos_loss,
                "train/mass_loss": mass_loss,
            }, step=global_step)
        
        if config["training"]["clip_value"]:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), \
                                           config["training"]["clip_value"])
        optimizer.step()

    train_loss /= (i + 1)  # i + 1 since i starts at 0
    train_pos_loss /= (i + 1)
    train_mass_loss /= (i + 1)

    if epoch % config["logging"]["epoch_log_interval"] == 0 and rank ==0:
         logging.info(f"Epoch {epoch}, Total: {train_loss:.6f}, Pos: {pos_loss:.6f}, Mass: {mass_loss:.6f}") 

    return train_loss, global_step

def valid_epoch(ddp_model, valid_dl, device, config, epoch, rank, global_step): 
    
    ddp_model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        for i, samples in enumerate(valid_dl):
            samples = samples.to(device)
            loss, pos_loss, mass_loss = ddp_model(samples=samples)
            
            wandb.log({"valid/batch_loss": loss.item()}, \
                  step=global_step) if config["log_wandb"] and rank == 0 else None
            if config["log_wandb"] and rank == 0:
                wandb.log({
                    "valid/batch_loss": loss.item(),
                    "valid/pos_loss": pos_loss,
                    "valid/mass_loss": mass_loss,
                }, step=global_step)

            valid_loss += loss.item()
    
    valid_loss /= (i + 1)   # i + 1 since i starts at 0

    if epoch % config["logging"]["epoch_log_interval"] == 0 and rank == 0:
        logging.info(f"Epoch {epoch}, Validation loss: {valid_loss}")

    return valid_loss

def train_model(config, output_dir, rank, local_rank):
    """Main training loop."""

    # Initialize training
    train_dl, valid_dl, train_sampler, ddp_model, optimizer, decay_scheduler, start_epoch = initialize_training(config, rank, local_rank)
    
    # Initialize trackers for early stopping
    best_loss = float('inf')
    patience_counter = 0

    # Track global step across epochs for wandb
    global_step = 0  

    # Learning rate at start of training
    if rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]  # get last lr
        wandb.log({"lr": current_lr}, step=global_step) if config["log_wandb"] else None

    # Track gradients and params
    wandb.watch(ddp_model, log="all", log_freq=config["wandb"]["log_freq"]) \
        if config["log_wandb"] and rank == 0 else None

    if rank == 0:
        logging.info("Started training")

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        
        # Randomly shuffle dataset before dividing it into chunks to each GPU
        train_sampler.set_epoch(epoch)

        train_loss, global_step = train_epoch(ddp_model, train_dl, optimizer, \
                                              local_rank, config, epoch, rank, global_step)
        if rank == 0:
            valid_loss = valid_epoch(ddp_model, valid_dl, local_rank, config, epoch, rank, global_step)
            wandb.log({"train/loss": train_loss, "valid/loss": valid_loss, "epoch": epoch}, \
                      step=global_step) if config["log_wandb"] else None
            
            # Decay scheduler
            if config["training"]["scheduler"]["type"] == "cosine": 
                decay_scheduler.step()
            if config["training"]["scheduler"]["type"] == "plateau":
                decay_scheduler.step(valid_loss)
        
            # Early stopping
            best_loss, patience_counter = evaluate_early_stopping(
                valid_loss, best_loss, patience_counter, config["training"]["early_stopping"]["min_delta"])
            if patience_counter >= config["training"]["early_stopping"]["patience"] and rank == 0 and epoch > 500:
                logging.info(f"Early stopping triggered at epoch {epoch}. Saving checkpoint.")
                file_path = save_model(ddp_model, f"model_early_stopping_epoch_{epoch}.pth", optimizer, 
                                    decay_scheduler, epoch, output_dir)
                if config["log_wandb"]:    
                    artifact = wandb.Artifact("model", type="model")
                    artifact.add_file(file_path)
                    wandb.run.log_artifact(artifact)

                # DDP 
                cleanup()

                wandb.finish() if config["log_wandb"] else None
                break

        # Warmup scheduler
        adjust_learning_rate(optimizer, epoch, config) 

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]  # get last lr
            wandb.log({"lr": current_lr}, step=global_step) if config["log_wandb"] else None
        
        # Save model periodically
        if epoch % config["logging"]["model_save_interval"] == 0 and rank == 0:
            file_path = save_model(ddp_model, f"model_epoch_{epoch}.pth", optimizer, 
                                   decay_scheduler, epoch, output_dir)
            if config["log_wandb"]:
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(file_path)
                wandb.run.log_artifact(artifact)

    # Finished training
    if rank == 0 and patience_counter < config["training"]["early_stopping"]["patience"]: 
        logging.info("Finished training")
        file_path = save_model(ddp_model, f"model_final.pth", optimizer, 
                            decay_scheduler, epoch, output_dir)
        if config["log_wandb"]:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(file_path)
            wandb.run.log_artifact(artifact)
        # DDP 
        cleanup()

        wandb.finish() if config["log_wandb"] and rank == 0 else None

def main():
    args = parse_arguments()
    yaml_config = load_config(args.config)
    final_config = merge_configs(yaml_config, args) # CLI overrides YAML

    # Generate unique directiory with time stamp and save config file there
    output_dir = unique_output_dir(final_config) 
    copy_config_to_output(args.config, output_dir)
    setup_logging(final_config, output_dir)

    if not torch.cuda.is_available():
        raise Exception('CUDA not found')
    
    ######################## DDP ##############################
    gpus_per_node = torch.cuda.device_count()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    if rank == 0:
        logging.info(f"Output_dir: {output_dir}")
        logging.info("Final training configuration:\n%s", pprint.pformat(final_config))
    
    train_model(final_config, output_dir, rank, local_rank)

if __name__ == "__main__":
    main()