import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / 'code'))

from common.utils import load_config
from common.plot_utils import setup_plot_style, get_figure_dir

def plot_part1(data_dir, fig_dir):
    setup_plot_style()
    
    # Avalanche sizes
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        gauss = np.load(data_dir / "part1_gaussian.npz")
        unif = np.load(data_dir / "part1_uniform.npz")
        
        ax.loglog(gauss['size_centers'], gauss['size_probs'], 'o', label='Gaussian', markersize=4, alpha=0.7)
        
        ax.loglog(unif['size_centers'], unif['size_probs'], 's', label='Uniform', markersize=4, alpha=0.7)
        
        # Branching process reference (size tau = 1.5)
        xs = np.logspace(0, np.log10(max(gauss['size_centers'].max(), unif['size_centers'].max())), 100)
        
        # Offset to place the reference line above the data
        y0 = gauss['size_probs'][0] if len(gauss['size_probs']) > 0 else 0.1
        ys = 10 * y0 * (xs / xs[0])**(-1.5)
        
        ax.loglog(xs, ys, 'k--', label=r'BP Theory $\tau=1.5$')
        
        ax.set_xlabel('Avalanche Size $A$')
        ax.set_ylabel('Probability $P(A)$')
        ax.set_title('Part 1: Avalanche Size Distribution')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "task_15_part1_sizes.pdf")
        
    except FileNotFoundError:
        print("Part 1 data not found. Run run_part1.py first.")
    finally:
        plt.close(fig)

    # Lifetimes
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        ax.loglog(gauss['lifetime_centers'], gauss['lifetime_probs'], 'o', label='Gaussian', markersize=4, alpha=0.7)
        
        ax.loglog(unif['lifetime_centers'], unif['lifetime_probs'], 's', label='Uniform', markersize=4, alpha=0.7)
        
        # Branching process reference (lifetime tau = 2.0)
        xs = np.logspace(0, np.log10(max(gauss['lifetime_centers'].max(), unif['lifetime_centers'].max())), 100)
        y0 = gauss['lifetime_probs'][0] if len(gauss['lifetime_probs']) > 0 else 0.5
        ys = 10 * y0 * (xs / xs[0])**(-2.0)
        ax.loglog(xs, ys, 'k--', label=r'BP Theory $\tau_t=2.0$')
        
        ax.set_xlabel('Lifetime $T$')
        ax.set_ylabel('Probability $P(T)$')
        ax.set_title('Part 1: Avalanche Lifetime Distribution')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "task_15_part1_lifetimes.pdf")
        
    except FileNotFoundError:
        pass
    finally:
        plt.close(fig)

def plot_part2(config, data_dir, fig_dir):
    setup_plot_style()
    gammas = config['part2']['gammas']
    
    # Avalanche sizes
    fig, ax = plt.subplots(figsize=(8, 6))
    max_x = 1
    colors = plt.cm.viridis(np.linspace(0, 1, len(gammas)))
    
    try:
        for i, gamma in enumerate(gammas):
            data = np.load(data_dir / f"part2_gamma_{gamma}.npz")
            lbl = r'$\gamma = \infty$' if gamma >= 100 else rf'$\gamma = {gamma}$'
            ax.loglog(data['size_centers'], data['size_probs'], 'o-', label=lbl, markersize=3, alpha=0.8, color=colors[i])
            max_x = max(max_x, data['size_centers'].max())
            
        xs = np.logspace(0, np.log10(max_x), 100)
        y0 = data['size_probs'][0] if 'data' in locals() and len(data['size_probs']) > 0 else 0.1
        ys = 10 * y0 * (xs / xs[0])**(-1.5)
        ax.loglog(xs, ys, 'k--', label=r'BP Theory $\tau=1.5$', linewidth=2)
        
        ax.set_xlabel('Avalanche Size $A$')
        ax.set_ylabel('Probability $P(A)$')
        ax.set_title('Part 2: Avalanche Size Distribution (Scale-Free)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "task_15_part2_sizes.pdf")
    except FileNotFoundError as e:
        print(f"Part 2 data not found: {e}")
    finally:
        plt.close(fig)
        
    # Lifetimes
    fig, ax = plt.subplots(figsize=(8, 6))
    max_x = 1
    
    try:
        for i, gamma in enumerate(gammas):
            data = np.load(data_dir / f"part2_gamma_{gamma}.npz")
            lbl = r'$\gamma = \infty$' if gamma >= 100 else rf'$\gamma = {gamma}$'
            ax.loglog(data['lifetime_centers'], data['lifetime_probs'], 'o-', label=lbl, markersize=3, alpha=0.8, color=colors[i])
            max_x = max(max_x, data['lifetime_centers'].max())
            
        xs = np.logspace(0, np.log10(max_x), 100)
        y0 = data['lifetime_probs'][0] if 'data' in locals() and len(data['lifetime_probs']) > 0 else 0.5
        ys = 10 * y0 * (xs / xs[0])**(-2.0)
        ax.loglog(xs, ys, 'k--', label=r'BP Theory $\tau_t=2.0$', linewidth=2)
        
        ax.set_xlabel('Lifetime $T$')
        ax.set_ylabel('Probability $P(T)$')
        ax.set_title('Part 2: Avalanche Lifetime Distribution (Scale-Free)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "task_15_part2_lifetimes.pdf")
    except FileNotFoundError:
        pass
    finally:
        plt.close(fig)


def main():
    config = load_config("task_15")
    data_dir = base_dir / "data" / "task_15"
    fig_dir = get_figure_dir("task_15")
    
    print("Plotting Part 1...")
    plot_part1(data_dir, fig_dir)
    
    print("Plotting Part 2...")
    plot_part2(config, data_dir, fig_dir)
    
    print("Done! Plots saved in latex/figures/task_15/")

if __name__ == "__main__":
    main()
