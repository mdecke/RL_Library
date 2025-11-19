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
    
    # Preserve step information if available
    if "step" in df.columns:
        first_cycle = df[cycle_col].dropna().unique()[0]
        out["step"] = df[df[cycle_col] == first_cycle]["step"].dropna().reset_index(drop=True)
    
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

def plot_action_predictions(true_actions, predicted_actions, action_dim=8, plots_dir=None):
    """Plot true vs predicted actions for each dimension."""
    n_cols = 4
    n_rows = (action_dim + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    axes = axes.flatten()
    
    # Calculate overall metrics
    overall_mse = ((true_actions - predicted_actions) ** 2).mean()
    per_dim_mses = []
    
    for i in range(action_dim):
        ax = axes[i]
        steps = np.arange(len(true_actions))
        ax.plot(steps, true_actions[:, i], label='True', alpha=0.7, linewidth=1)
        ax.plot(steps, predicted_actions[:, i], label='Predicted', alpha=0.7, linewidth=1)
        
        # Calculate per-dimension MSE
        dim_mse = ((true_actions[:, i] - predicted_actions[:, i]) ** 2).mean()
        per_dim_mses.append(dim_mse)
        
        ax.set_title(f'Action dim {i} (MSE: {dim_mse:.4f})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Action value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for i in range(action_dim, len(axes)):
        axes[i].axis('off')
    
    avg_mse_per_dim = np.mean(per_dim_mses)
    
    plt.suptitle(f'One-step Action Predictions | Overall MSE: {overall_mse:.4f} | Avg MSE/dim: {avg_mse_per_dim:.4f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if plots_dir:
        import os
        save_path = os.path.join(plots_dir, 'action_predictions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_rollout_rewards(all_instantaneous_rewards, all_cumulative_rewards, plots_dir=None):
    """Plot instantaneous and cumulative rewards from rollout evaluation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot instantaneous rewards
    ax = axes[0]
    for i, rewards in enumerate(all_instantaneous_rewards):
        steps = np.arange(len(rewards))
        ax.plot(steps, rewards, alpha=0.6, label=f'Episode {i+1}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Instantaneous Reward')
    ax.set_title('Instantaneous Rewards per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot cumulative rewards
    ax = axes[1]
    for i, cum_rewards in enumerate(all_cumulative_rewards):
        steps = np.arange(len(cum_rewards))
        ax.plot(steps, cum_rewards, alpha=0.6, label=f'Episode {i+1}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Rewards per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Expert Policy Rollout Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if plots_dir:
        save_path = os.path.join(plots_dir, 'expert_rollout_evaluation.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved rollout evaluation plot to {save_path}")
    
    plt.show()
    return fig


def main():
    args = parse_args()

    # Load both guided and regular training data
    guided_stats_folder = os.path.join(args.log_dir, args.task, args.algorithm, "guided_training_stats")
    regular_stats_folder = os.path.join(args.log_dir, args.task, args.algorithm, "training_stats")
    
    has_guided = os.path.exists(guided_stats_folder) and any(f.endswith('.csv') for f in os.listdir(guided_stats_folder))
    has_regular = os.path.exists(regular_stats_folder) and any(f.endswith('.csv') for f in os.listdir(regular_stats_folder))
    
    if not has_guided and not has_regular:
        print(f"[ERROR] No training data found in {guided_stats_folder} or {regular_stats_folder}")
        return
    
    title = f"{args.algorithm} · {args.task}"
    smoothing = max(int(args.smoothing_window), 0)
    
    save_root = os.path.join(args.save_dir, args.task, args.algorithm)
    os.makedirs(save_root, exist_ok=True)

    # Process guided training data if available
    guided_returns_stats = None
    guided_policy_stats = None
    guided_q_stats = None
    
    if has_guided:
        guided_data = make_data_frame(guided_stats_folder)
        if "seed" not in guided_data.columns:
            guided_data["seed"] = 0
        
        guided_returns_stats = compute_cycle_stats(guided_data, "seed", "return")
        guided_policy_stats = compute_cycle_stats(guided_data, "seed", "policy_loss")
        guided_q_stats = compute_cycle_stats(guided_data, "seed", "q_loss")

        if smoothing > 1:
            guided_returns_stats = apply_smoothing(guided_returns_stats, smoothing)
            guided_policy_stats = apply_smoothing(guided_policy_stats, smoothing)
            guided_q_stats = apply_smoothing(guided_q_stats, smoothing)
    
    # Process regular training data if available
    regular_returns_stats = None
    regular_policy_stats = None
    regular_q_stats = None
    
    if has_regular:
        regular_data = make_data_frame(regular_stats_folder)
        if "seed" not in regular_data.columns:
            regular_data["seed"] = 0
        
        regular_returns_stats = compute_cycle_stats(regular_data, "seed", "return")
        regular_policy_stats = compute_cycle_stats(regular_data, "seed", "policy_loss")
        regular_q_stats = compute_cycle_stats(regular_data, "seed", "q_loss")

        if smoothing > 1:
            regular_returns_stats = apply_smoothing(regular_returns_stats, smoothing)
            regular_policy_stats = apply_smoothing(regular_policy_stats, smoothing)
            regular_q_stats = apply_smoothing(regular_q_stats, smoothing)

    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    
    # Plot returns
    ax = axes[0]
    if regular_returns_stats is not None:
        x = regular_returns_stats["step"] if "step" in regular_returns_stats.columns else np.arange(len(regular_returns_stats["mean"]))
        ax.plot(x, regular_returns_stats["mean"], label="Regular Training", color='#66ccee', linewidth=1.5, linestyle='-')
        ax.fill_between(x, regular_returns_stats["lower"], regular_returns_stats["upper"], 
                        alpha=0.2, color='#66ccee')
    
    if guided_returns_stats is not None:
        x = guided_returns_stats["step"] if "step" in guided_returns_stats.columns else np.arange(len(guided_returns_stats["mean"]))
        ax.plot(x, guided_returns_stats["mean"], label="Expert-Guided Training", color='#ee6677', linewidth=1.5, linestyle='-.')
        ax.fill_between(x, guided_returns_stats["lower"], guided_returns_stats["upper"], 
                        alpha=0.2, color='#ee6677')
    
    ax.set_ylabel("Return")
    ax.set_xlabel("Training Steps")
    ax.set_title(f"{title} - Episodic Return Comparison")
    ax.axhline(y=0.0, linewidth=1, color="k", linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize='medium')
    
    # Plot policy loss
    ax = axes[1]
    if regular_policy_stats is not None:
        x = regular_policy_stats["step"] if "step" in regular_policy_stats.columns else np.arange(len(regular_policy_stats["mean"]))
        ax.plot(x, regular_policy_stats["mean"], label="Regular Training", color='#66ccee', linewidth=1.5, linestyle='-')
        ax.fill_between(x, regular_policy_stats["lower"], regular_policy_stats["upper"], 
                        alpha=0.2, color='#66ccee')
    
    if guided_policy_stats is not None:
        x = guided_policy_stats["step"] if "step" in guided_policy_stats.columns else np.arange(len(guided_policy_stats["mean"]))
        ax.plot(x, guided_policy_stats["mean"], label="Expert-Guided Training", color="#ee6677", linewidth=1.5, linestyle='-.')
        ax.fill_between(x, guided_policy_stats["lower"], guided_policy_stats["upper"], 
                        alpha=0.2, color="#ee6677")
    
    ax.set_ylabel("Policy Loss")
    ax.set_xlabel("Training Steps")
    ax.set_title(f"{title} - Policy Loss Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize='medium')
    
    # Plot Q loss
    ax = axes[2]
    if regular_q_stats is not None:
        x = regular_q_stats["step"] if "step" in regular_q_stats.columns else np.arange(len(regular_q_stats["mean"]))
        ax.plot(x, regular_q_stats["mean"], label="Regular Training", color='#66ccee', linewidth=1.5, linestyle='-')
        ax.fill_between(x, regular_q_stats["lower"], regular_q_stats["upper"], 
                        alpha=0.2, color='#66ccee')
    
    if guided_q_stats is not None:
        x = guided_q_stats["step"] if "step" in guided_q_stats.columns else np.arange(len(guided_q_stats["mean"]))
        ax.plot(x, guided_q_stats["mean"], label="Expert-Guided Training", color='#ee6677', linewidth=1.5, linestyle='-.')
        ax.fill_between(x, guided_q_stats["lower"], guided_q_stats["upper"], 
                        alpha=0.2, color='#ee6677')
    
    ax.set_ylabel("Q Loss")
    ax.set_xlabel("Training Steps")
    ax.set_title(f"{title} - Q-function Loss Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize='medium')
    
    smoothing_text = f" (smoothed={smoothing})" if smoothing > 1 else ""
    fig.suptitle(f"{title} - Training Comparison{smoothing_text}", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    out_path = os.path.join(save_root, "training_comparison.svg")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved comparison plot → {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
