import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base_classes.utils import load_file, make_data_frame

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training metrics for a given environment and algorithm")

    p.add_argument("--task", type=str, default="Pendulum-v1", help="Gym environment name")
    p.add_argument("--algorithm", type=str, default="TD3", help="RL algorithm used")
    p.add_argument("--log_dir", type=str, default="./logs", help="Base directory where logs are stored (expects subdirs task/algorithm)")
    p.add_argument("--smoothing_window", type=int, default=0,
                   help="Window for moving-average smoothing of mean/std bands (0 or 1 = off)")
    p.add_argument("--save_dir", type=str, default="./plots", help="Directory to save plots")
    p.add_argument("--show", action="store_true", help="Show the figure interactively")

    return p.parse_args()

def smooth_array(x: np.ndarray, window: int) -> np.ndarray:
    if window is None or window < 2:
        return x
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def compute_cycle_stats(df: pd.DataFrame, cycle_col: str, value_col: str) -> pd.DataFrame:
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in CSV. Available: {list(df.columns)}")
    if cycle_col not in df.columns:
        raise KeyError(f"Cycle column '{cycle_col}' not found in CSV. Available: {list(df.columns)}")

    out = pd.DataFrame()
    for cycle in df[cycle_col].dropna().unique():
        series = (df[df[cycle_col] == cycle][value_col].dropna().reset_index(drop=True))
        out[f"cycle {cycle}"] = series

    # stats across cycles (row-wise)
    out["mean"] = out.mean(axis=1, skipna=True)
    out["std"] = out.std(axis=1, skipna=True)
    out["upper"] = out["mean"] + out["std"]
    out["lower"] = out["mean"] - out["std"]
    return out


def apply_smoothing(stats_df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window is None or window < 2:
        return stats_df
    sm = stats_df.copy()
    for col in ("mean", "upper", "lower"):
        sm[col] = smooth_array(sm[col].to_numpy(), window)
    return sm


def plot_returns(ax, stats_df: pd.DataFrame, smoothing_window: int) -> None:
    x = np.arange(len(stats_df["mean"]))
    ax.plot(x, stats_df["mean"], label="mean return")
    ax.fill_between(x, stats_df["lower"], stats_df["upper"], alpha=0.15)
    ax.set_ylabel("Return")
    ax.set_xlabel("Training Steps")
    ax.set_title(f"Mean Episodic Return (smoothed={smoothing_window if smoothing_window>1 else 'off'})")
    ax.axhline(y=0.0, linewidth=2, color="k")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize="x-small")


def plot_series(ax, stats_df: pd.DataFrame, ylabel: str, title: str, label: str) -> None:
    x = np.arange(len(stats_df["mean"]))
    ax.plot(x, stats_df["mean"], label=label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Training Steps")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize="x-small")

def create_subplot_grid(n_plots, figsize_per_subplot=(5, 4)):
    max_plots_per_fig = 6
    n_figures = int(np.ceil(n_plots / max_plots_per_fig))
    
    figs = []
    axes = []
    
    plot_idx = 0
    
    for _ in range(n_figures):
        remaining_plots = n_plots - plot_idx
        n_plots_this_fig = min(remaining_plots, max_plots_per_fig)

        if n_plots_this_fig <= 3:
            n_rows = n_plots_this_fig
            n_cols = 1
        else:
            n_rows = 2
            n_cols = 3

        figsize = (figsize_per_subplot[0] * n_cols, 
                   figsize_per_subplot[1] * n_rows)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_plots_this_fig == 1:
            ax = np.array([ax])
        else:
            ax = ax.flatten() if n_plots_this_fig > 1 else np.array([ax])
        
        for i in range(n_plots_this_fig, len(ax)):
            ax[i].set_visible(False)
        
        figs.append(fig)
        axes.append(ax[:n_plots_this_fig])
        plot_idx += n_plots_this_fig
    
    return figs, axes

def plot_prediction_accuracy(acts:np.ndarray, predictions:np.ndarray, action_dim:int) -> None: 
    figs, axes_list = create_subplot_grid(action_dim, figsize_per_subplot=(6, 4))
    action_idx = 0
    
    for fig_idx, (fig, axes) in enumerate(zip(figs, axes_list)):
        if len(figs) > 1:
            start_dim = fig_idx * 6
            end_dim = min(start_dim + len(axes), action_dim)
            fig.suptitle(f'Action Predictions - Figure {fig_idx + 1}/{len(figs)} '
                        f'(Dimensions {start_dim}-{end_dim-1})',
                        fontsize=16, fontweight='bold', y=1.00)
        
        # Plot each dimension in this figure
        for ax in axes:
            # Scatter plot of predictions vs true values
            ax.scatter(acts[:, action_idx], 
                      predictions[:, action_idx],
                      alpha=0.5, s=10, label='Predictions')
            
            # Perfect prediction line (y = ŷ)
            min_val = min(acts[:, action_idx].min().item(), 
                         predictions[:, action_idx].min().item())
            max_val = max(acts[:, action_idx].max().item(),
                         predictions[:, action_idx].max().item())
            ax.plot([min_val, max_val], [min_val, max_val],
                   'r--', linewidth=2, label='y = ŷ')
            
            # Labels and title
            ax.set_xlabel(f'True Action {action_idx}', fontsize=12)
            ax.set_ylabel(f'Predicted Action {action_idx}', fontsize=12)
            ax.set_title(f'Action Dimension {action_idx}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate metrics
            mse = ((acts[:, action_idx] - predictions[:, action_idx])**2).mean().item()
            ss_res = ((acts[:, action_idx] - predictions[:, action_idx])**2).sum().item()
            ss_tot = ((acts[:, action_idx] - acts[:, action_idx].mean())**2).sum().item()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Add metrics text box
            ax.text(0.95, 0.95, f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            action_idx += 1
        
        plt.figure(fig.number)
        plt.tight_layout()
    
    return figs

def plot_trajectories(data:np.ndarray, label:str) -> None:
    data_dim = data.shape[1]
    figs, axes_list = create_subplot_grid(data_dim, figsize_per_subplot=(6, 4))

    dim_idx = 0

    for fig_idx, (fig, axes) in enumerate(zip(figs, axes_list)):
        if len(figs) > 1:
            start_dim = fig_idx * 6
            end_dim = min(start_dim + len(axes), data_dim)
            fig.suptitle(f'{label} RollOut - Figure {fig_idx + 1}/{len(figs)} '
                        f'(Dimensions {start_dim}-{end_dim-1})',
                        fontsize=16, fontweight='bold', y=1.00)
        
        # Plot each dimension in this figure
        for ax in axes:
            ax.plot(data[:, dim_idx], alpha=0.7)

            ax.set_xlabel('Env steps', fontsize=12)
            ax.set_ylabel(f'{label} {dim_idx}', fontsize=12)
            ax.set_title(f'{label} Dimension {dim_idx}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            dim_idx += 1

        plt.figure(fig.number)
        plt.tight_layout()
    
    return figs


def main():
    args = parse_args()

    training_stats_folder = os.path.join(args.log_dir, args.task, args.algorithm, "training_stats")
    training_data = make_data_frame(training_stats_folder)
    
    title = f"{args.algorithm} · {args.task}"
    smoothing = max(int(args.smoothing_window), 0)

    # Compute stats for each metric (assuming single cycle/run for now)
    # Add a dummy cycle column if it doesn't exist
    if "seed" not in training_data.columns:
        training_data["seed"] = 0
    
    returns_stats = compute_cycle_stats(training_data, "seed", "return")
    policy_stats = compute_cycle_stats(training_data, "seed", "policy_loss")
    q_stats = compute_cycle_stats(training_data, "seed", "q_loss")

    if smoothing > 1:
        returns_stats = apply_smoothing(returns_stats, smoothing)
        policy_stats = apply_smoothing(policy_stats, smoothing)
        q_stats = apply_smoothing(q_stats, smoothing)

    save_root = os.path.join(args.save_dir, args.task, args.algorithm)
    os.makedirs(save_root, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    
    plot_returns(axes[0], returns_stats, smoothing)
    axes[0].set_title(f"{title} - Mean Episodic Return")

    plot_series(
        axes[1],
        policy_stats,
        ylabel="Policy Loss",
        title=f"{title} - Mean Policy Loss (smoothed={smoothing if smoothing>1 else 'off'})",
        label="policy loss",
    )
    
    plot_series(
        axes[2],
        q_stats,
        ylabel="Value Loss",
        title=f"{title} - Mean Q-function Loss (smoothed={smoothing if smoothing>1 else 'off'})",
        label="q loss",
    )
    
    out_path = os.path.join(save_root, "all_metrics.svg")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved plot → {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
