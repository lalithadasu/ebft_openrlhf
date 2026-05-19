"""Print all sweep configs with their array indices."""
from itertools import product
from omegaconf import OmegaConf
import sys

KEY_ABBREVS = {
    "actor_learning_rate": "a_lr", "pretrain": "pt", "critic_pretrain": "cpt",
    "ce_loss_coef": "ce", "rl_loss_coef": "rt", "diversity_rew_coef": "pr",
    "use_whitening": "wh", "num_episodes": "ne", "occupancy_distance": "occ_dist",
    "occupancy_reward_mode": "occ",
}

def flatten(config):
    flat = {}
    for k, v in config.items():
        if isinstance(v, dict) or OmegaConf.is_dict(v):
            for k2, v2 in flatten(v).items():
                flat[f"{k}.{k2}"] = v2
        else:
            flat[k] = v
    return flat

def grid_to_list(grid):
    flat_grid = flatten(grid)
    iter_overwrites, flat_overwrites = {}, {}
    for k, v in flat_grid.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            iter_overwrites[k] = v
        else:
            flat_overwrites[k] = v
    grid_list = []
    for values in product(*iter_overwrites.values()):
        d = dict(zip(iter_overwrites.keys(), values))
        d.update(flat_overwrites)
        grid_list.append(d)
    return grid_list

sweep_config = sys.argv[1] if len(sys.argv) > 1 else "configs/qa_code.yaml"
base = OmegaConf.load(sweep_config)
sweeps = base.pop("sweep")

config_list = []
for sweep in sweeps:
    merged = OmegaConf.merge(base, sweep)
    config_list += grid_to_list(merged)

print(f"Total configs: {len(config_list)}\n")

interesting = ["pretrain", "use_whitening", "ce_loss_coef", "diversity_rew_coef",
               "num_episodes", "occupancy_reward_mode", "occupancy_distance"]

for i, cfg in enumerate(config_list):
    parts = {k: cfg[k] for k in interesting if k in cfg}
    print(f"[{i:3d}]  {parts}")
