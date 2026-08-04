import yaml
import numpy as np
import random
from pathlib import Path

def load_config(task_name="task_15"):
    base_dir = Path(__file__).parent.parent.parent
    
    with open(base_dir / "config" / "shared.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    with open(base_dir / "config" / task_name / "config.yaml", "r") as f:
        task_config = yaml.safe_load(f)
        
    config.update(task_config)
    return config

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
