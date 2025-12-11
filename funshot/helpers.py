import re
import yaml

def load_config(name: str):
    path = f"config/{name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
    raise IOError(f"Failed to load config at '{path}'")

