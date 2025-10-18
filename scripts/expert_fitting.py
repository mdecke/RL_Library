import os

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

from agents import create_expert
from base_classes.utils import load_config, make_data_frame
from plot import plot_prediction_accuracy


parser = argparse.ArgumentParser(description="Fit an expert model to data.")
parser.add_argument("--expert_type", type=str, choices=["mle","gmm","cnf"], default="mle", help="Type of expert model to fit")
parser.add_argument("--expert_domain", type=str, choices=["time","frequency"], default="time", help="Domain in which to fit the expert model")
parser.add_argument("--expert_data_path", type=str, required=True, help="Path to the training data (CSV format)")
parser.add_argument("--path_to_saved_expert", type=str, help="Directory to save the trained expert model")
parser.add_argument("--path_to_figures", type=str, help="Directory to save the prediction accuracy figures")
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

    if not os.path.exists(args.path_to_saved_expert):
        os.makedirs(args.path_to_saved_expert, exist_ok=True)

    expert_data = make_data_frame(args.expert_data_path)
    obss_tensor = torch.tensor(expert_data[[col for col in expert_data.columns if "obs" in col]].values, dtype=torch.float32).to(args.device)
    acts_tensor = torch.tensor(expert_data[[col for col in expert_data.columns if "act" in col]].values, dtype=torch.float32).to(args.device)

    cfg["obs_dim"] = obss_tensor.shape[1]
    cfg["action_dim"] = acts_tensor.shape[1]

    index = np.arange(len(expert_data))
    np.random.shuffle(index)

    test_idx = index[:int(cfg["test_split"] * len(index))]
    val_idx = index[int(cfg["test_split"] * len(index)):int(cfg["validation_split"] * len(index))+int(cfg["test_split"] * len(index))]
    train_idx = index[int(cfg["validation_split"] * len(index))+int(cfg["test_split"] * len(index)):]


    train_obss = obss_tensor[train_idx]
    train_acts = acts_tensor[train_idx]
    test_obss = obss_tensor[test_idx]
    test_acts = acts_tensor[test_idx]
    val_obss = obss_tensor[val_idx]
    val_acts = acts_tensor[val_idx]


    train_data = TensorDataset(train_obss, train_acts)
    train_loader = DataLoader(train_data, batch_size=cfg[f"{args.expert_type}"]["batch_size"], shuffle=True)
    val_data = TensorDataset(val_obss, val_acts)
    val_loader = DataLoader(val_data, shuffle=False)

    test_data = TensorDataset(test_obss, test_acts)
    test_loader = DataLoader(test_data, shuffle=False)

    
    expert = create_expert(args.expert_type, cfg)
    train_losses, val_losses = expert.train(train_loader, val_loader)
    test_loss = expert.validate(test_loader)
    expert.save(args.path_to_saved_expert)

    expert.model.eval()
    with torch.no_grad():
        if expert.preprocess_inputs:
            test_obss = expert.obs_preprocessor(test_obss).to(args.device)
        predicted_actions = expert.model.most_likely_component(test_obss)
    
    plots_dir = os.path.join(args.path_to_figures, args.expert_type)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
    figures = plot_prediction_accuracy(test_acts.to("cpu").numpy(), predicted_actions.to("cpu").numpy(), test_acts.shape[1])

    for fig_idx, fig in enumerate(figures):
        filename = f"action_predictions_fig{fig_idx+1}.png" if len(figures) > 1 else "action_predictions.png"
        fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved {filename}")

    plt.show()  # Show all figures


if __name__ == "__main__":
    main()
