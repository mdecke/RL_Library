import os
import yaml
import ast
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

from base_classes.models import MLE
from base_classes.utils import load_config, make_data_frame, gaussian_nll_loss

EXPERTS = {
    "mle": MLE,
    # "cnf": CNF,  # Placeholder for future expert models
}

parser = argparse.ArgumentParser(description="Fit an expert model to data.")
parser.add_argument("--expert_type", type=str, choices=["mle","cnf"], default="mle", help="Type of expert model to fit")
parser.add_argument("--expert_domain", type=str, choices=["time","frequency"], default="time", help="Domain in which to fit the expert model")
parser.add_argument("--expert_data_path", type=str, required=True, help="Path to the training data (CSV format)")
parser.add_argument("--path_to_saved_expert", type=str, help="Directory to save the trained expert model")
parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

args = parser.parse_args()

def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_file = os.path.join("configs", "ExpertConfig.yaml")
    cfg = load_config(config_file, args)
    cfg['expert_type'] = args.expert_type
    cfg['expert_domain'] = args.expert_domain
  
    expert_data = make_data_frame(args.expert_data_path)
    obss_tensor = torch.tensor(expert_data[[col for col in expert_data.columns if "obs" in col]].values, dtype=torch.float32).to(args.device)
    acts_tensor = torch.tensor(expert_data[[col for col in expert_data.columns if "act" in col]].values, dtype=torch.float32).to(args.device)

    index = np.arange(len(expert_data))
    np.random.shuffle(index)
    print(index)
    test_idx = index[:int(cfg["test_split"] * len(index))]
    val_idx = index[int(cfg["test_split"] * len(index)):int(cfg["validation_split"] * len(index))]
    train_idx = index[int(cfg["validation_split"] * len(index)):]

    train_obss = obss_tensor[train_idx]
    train_acts = acts_tensor[train_idx]
    test_obss = obss_tensor[test_idx]
    test_acts = acts_tensor[test_idx]
    val_obss = obss_tensor[val_idx]
    val_acts = acts_tensor[val_idx]

    train_data = TensorDataset(train_obss, train_acts)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
    val_data = TensorDataset(val_obss, val_acts)
    val_loader = DataLoader(val_data, shuffle=False)

    test_data = TensorDataset(test_obss, test_acts)
    test_loader = DataLoader(test_data, shuffle=False)

   



if __name__ == "__main__":
    main()
