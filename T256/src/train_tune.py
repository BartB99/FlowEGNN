"""
Hyperparameter tuning with Ray Tune for T256-SUBBOX.
Runs parallel single-GPU training trials.

Usage:
    python src/train_tune.py --config Configs/overfit_configs.yaml

Tuning targets: coords_weight, hidden_nf, sigma_0, lambda_mass, coord_agg
Fixed: n_layers=4, r=0.3, latent_nf=64, batch_size=16, OT, cosine schedule, 
       gradient clipping=1.0, attention=True, recurrent=True, layer norm
"""

import copy
import logging
import os
import sys
import yaml
from argparse import ArgumentParser

import ray
import torch
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

# ---------- Search space ----------
SEARCH_SPACE = {
    "coords_weight": tune.choice([0.5, 1.0, 2.0]),
    "hidden_nf":     tune.choice([128, 192, 256]),
    "lambda_mass":   tune.choice([0.1, 0.5, 1.0, 2.0]),
    "lr":            tune.loguniform(1e-4, 5e-3),
    "norm":          tune.choice(["layer", "weight", "none"]),
}


def train_one_trial(ray_config, base_config_path, tune_epochs, project_root):
    """Train a single model with one set of hyperparameters."""

    sys.path.insert(0, os.path.join(project_root, "src"))
    os.chdir(project_root)

    from egnn import EGNN
    from fm import FlowMatching
    from dataset import create_dataloader
    from utils import load_config, get_activation_fn, get_scheduler, adjust_learning_rate

    logging.disable(logging.CRITICAL)

    # Load base config and override with this trial's hyperparameters
    config = copy.deepcopy(load_config(base_config_path))

    config["model"]["egnn"]["coords_weight"]   = ray_config["coords_weight"]
    config["model"]["egnn"]["hidden_nf"]        = ray_config["hidden_nf"]
    config["model"]["fm"]["lambda_mass"]        = ray_config["lambda_mass"]
    config["model"]["egnn"]["norm"]             = ray_config["norm"] if ray_config["norm"] != "none" else None
    config["training"]["scheduler"]["initial_lr"] = ray_config["lr"]

    # Training settings for tuning
    config["training"]["epochs"]    = tune_epochs
    config["log_wandb"]             = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config["training"]["batch_size"]

    # --- Data loading (non-distributed) ---
    train_kwargs = {"batch_size": batch_size, "num_workers": 2, "pin_memory": True}
    valid_kwargs = {"batch_size": config["training"]["batch_size_valid"], "num_workers": 2, "pin_memory": True}

    train_dl, valid_dl, _ = create_dataloader(
        cosmologies_dir=config["data"]["cosmologies_dir"],
        cosmologies_info_dir=config["data"]["cosmologies_info_dir"],
        distributed=False,
        train_kwargs=train_kwargs,
        valid_kwargs=valid_kwargs,
        rotations=config["data"]["rotations"],
        translations=config["data"]["translations"],
        cosm_param=config["data"]["cosm_param"],
    )

    # --- Model ---
    input_theta_d = len(config["data"]["cosm_param"]) if config["data"]["cosm_param"] is not None else 5

    egnn = EGNN(
        t_embed_dim=config["model"]["fm"]["t_embed_dim"],
        input_node_d=config["model"]["egnn"]["input_node_d"],
        input_theta_d=input_theta_d,
        theta_param_embd_dim=config["model"]["egnn"]["theta_param_embd_dim"],
        hidden_nf=ray_config["hidden_nf"],
        latent_nf=config["model"]["egnn"]["latent_nf"],
        theta_nf=config["model"]["egnn"]["theta_nf"],
        n_layers=config["model"]["egnn"]["n_layers"],
        mlp_layers=config["model"]["egnn"]["mlp_layers"],
        single_layer=config["model"]["egnn"]["single_layer"],
        recurrent=config["model"]["egnn"]["recurrent"],
        activation=get_activation_fn(config["model"]["egnn"]["activation"]),
        norm=ray_config["norm"],
        attention=config["model"]["egnn"]["attention"],
        scale_pred=config["model"]["egnn"]["scale_pred"],
        coords_weight=ray_config["coords_weight"],
        norm_diff=config["model"]["egnn"]["norm_diff"],
    )

    model = FlowMatching(
        sigma_0=config["model"]["fm"]["sigma_0"],
        sigma_sched=config["model"]["fm"]["sigma_sched"],
        t_embed_dim=config["model"]["fm"]["t_embed_dim"],
        version=config["model"]["fm"]["version"],
        vnet=egnn,
        batch_size=batch_size,
        prior=config["model"]["fm"]["prior"],
        k=config["model"]["fm"]["k"],
        r=config["model"]["fm"]["r"],
        optimal_transport=config["model"]["fm"]["optimal_transport"],
        t_embed=config["model"]["fm"]["t_embed"],
        lambda_mass=ray_config["lambda_mass"],
        dim=3,
    ).to(device)

    # --- Optimizer & scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=float(ray_config["lr"]),
        weight_decay=config["training"]["optimizer"]["weight_decay"],
    )
    decay_scheduler = get_scheduler(optimizer, config)

    clip_value = config["training"]["clip_value"]
    warmup_epochs = config["training"]["scheduler"].get("warmup_epochs", 0)

    # --- Training loop ---
    try:
        for epoch in range(1, tune_epochs + 1):

            # Train
            model.train()
            train_loss = 0.0
            train_pos_loss = 0.0
            train_mass_loss = 0.0

            for i, samples in enumerate(train_dl):
                optimizer.zero_grad()
                samples = samples.to(device)
                loss, pos_loss, mass_loss = model(samples=samples)
                loss.backward()

                if clip_value:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()

                train_loss += loss.item()
                train_pos_loss += pos_loss
                train_mass_loss += mass_loss

            train_loss /= (i + 1)
            train_pos_loss /= (i + 1)
            train_mass_loss /= (i + 1)

            # Validate
            model.eval()
            valid_loss = 0.0
            valid_pos_loss = 0.0
            valid_mass_loss = 0.0

            with torch.no_grad():
                for i, samples in enumerate(valid_dl):
                    samples = samples.to(device)
                    loss, pos_loss, mass_loss = model(samples=samples)
                    valid_loss += loss.item()
                    valid_pos_loss += pos_loss
                    valid_mass_loss += mass_loss

            valid_loss /= (i + 1)
            valid_pos_loss /= (i + 1)
            valid_mass_loss /= (i + 1)

            # Scheduler
            adjust_learning_rate(optimizer, epoch, config)
            if decay_scheduler is not None:
                if config["training"]["scheduler"]["type"] == "cosine":
                    decay_scheduler.step()
                elif config["training"]["scheduler"]["type"] == "plateau":
                    decay_scheduler.step(valid_loss)

            # Report to Ray
            ray.tune.report({
                "valid_loss": valid_loss,
                "valid_pos_loss": valid_pos_loss,
                "valid_mass_loss": valid_mass_loss,
                "train_loss": train_loss,
                "train_pos_loss": train_pos_loss,
                "train_mass_loss": train_mass_loss,
                "epoch": epoch,
            })

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        ray.tune.report({
            "valid_loss": float("inf"),
            "train_loss": float("inf"),
            "epoch": epoch,
        })
        print("OOM — trial killed.")


# ---------- Main ----------
def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of trials to run")
    parser.add_argument("--tune_epochs", type=int, default=300,
                        help="Max epochs per trial")
    parser.add_argument("--grace_period", type=int, default=50,
                        help="Min epochs before ASHA can kill a trial")
    parser.add_argument("--gpus_per_trial", type=float, default=1.0)
    args = parser.parse_args()

    base_config_path = os.path.abspath(args.config)
    project_root = os.path.abspath(os.path.join(os.path.dirname(base_config_path), ".."))

    ray.init(ignore_reinit_error=True, _temp_dir="/tmp/ray")
    print(f"Ray sees {ray.available_resources().get('GPU', 0)} GPUs")

    scheduler = ASHAScheduler(
        metric="valid_loss",
        mode="min",
        max_t=args.tune_epochs,
        grace_period=args.grace_period,
        reduction_factor=2,
    )

    trainable = tune.with_parameters(
        train_one_trial,
        base_config_path=base_config_path,
        tune_epochs=args.tune_epochs,
        project_root=project_root,
    )
    trainable = tune.with_resources(trainable, {"gpu": args.gpus_per_trial, "cpu": 2})

    tuner = Tuner(
        trainable,
        param_space=SEARCH_SPACE,
        tune_config=TuneConfig(
            scheduler=scheduler,
            num_samples=args.num_samples,
        ),
    )
    results = tuner.fit()

    best = results.get_best_result("valid_loss", "min")
    print(f"\n===== Best trial =====")
    print(f"  valid_loss     : {best.metrics['valid_loss']:.6f}")
    print(f"  valid_pos_loss : {best.metrics.get('valid_pos_loss', 'N/A')}")
    print(f"  valid_mass_loss: {best.metrics.get('valid_mass_loss', 'N/A')}")
    print(f"  config         : {best.config}")

    output_path = os.path.join(project_root, "results", "tune_best_config.yaml")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(best.config, f)
    print(f"  saved to       : {os.path.abspath(output_path)}")

    ray.shutdown()


if __name__ == "__main__":
    main()