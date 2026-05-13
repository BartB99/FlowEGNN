import yaml

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