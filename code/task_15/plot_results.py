import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / 'code'))

from common.utils import load_config
from common.plot_utils import setup_plot_style, get_figure_dir

FONT_SIZE = 15
LEGEND_SIZE = 11
TITLE_SIZE = 13

def setup_font_sizes():
    plt.rcParams.update({
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 3,
        'ytick.labelsize': FONT_SIZE - 3,
        'legend.fontsize': LEGEND_SIZE,
        'axes.titlesize': TITLE_SIZE,
    })

def plot_part1(data_dir, fig_dir):
    setup_plot_style()
    setup_font_sizes()

    gauss = np.load(data_dir / "part1_gaussian.npz")
    unif = np.load(data_dir / "part1_uniform.npz")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Avalanche size (left panel) ---
    ax = axes[0]
    ax.loglog(gauss['size_centers'], gauss['size_probs'], 'o', label='Gaussian', markersize=4, alpha=0.7)
    ax.loglog(unif['size_centers'], unif['size_probs'], 's', label='Uniform', markersize=4, alpha=0.7)

    xs = np.logspace(0, np.log10(max(gauss['size_centers'].max(), unif['size_centers'].max())), 100)
    y0 = gauss['size_probs'][0] if len(gauss['size_probs']) > 0 else 0.1
    ys = 10 * y0 * (xs / xs[0])**(-1.5)
    ax.loglog(xs, ys, 'k--', label=r'BP Theory $\tau=1.5$')

    ax.set_xlabel('Avalanche Size $s$')
    ax.set_ylabel('Probability $P(s)$')
    ax.set_title('Avalanche Size Distribution')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # --- Avalanche lifetime (right panel) ---
    ax = axes[1]
    ax.loglog(gauss['lifetime_centers'], gauss['lifetime_probs'], 'o', label='Gaussian', markersize=4, alpha=0.7)
    ax.loglog(unif['lifetime_centers'], unif['lifetime_probs'], 's', label='Uniform', markersize=4, alpha=0.7)

    xs = np.logspace(0, np.log10(max(gauss['lifetime_centers'].max(), unif['lifetime_centers'].max())), 100)
    y0 = gauss['lifetime_probs'][0] if len(gauss['lifetime_probs']) > 0 else 0.1
    ys = 10 * y0 * (xs / xs[0])**(-2.0)
    ax.loglog(xs, ys, 'k--', label=r'BP Theory $z=2.0$')

    ax.set_xlabel('Lifetime $t$')
    ax.set_ylabel('Probability $P(t)$')
    ax.set_title('Avalanche Lifetime Distribution')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "task_15_part1.pdf")
    plt.close(fig)


def plot_part2(config, data_dir, fig_dir):
    setup_plot_style()
    setup_font_sizes()

    gammas = config['part2']['gammas']
    colors = plt.cm.viridis(np.linspace(0, 1, len(gammas)))

    datasets = {}
    for i, gamma in enumerate(gammas):
        datasets[gamma] = np.load(data_dir / f"part2_gamma_{gamma}.npz")

    max_size_x = max(datasets[g]['size_centers'].max() for g in gammas)
    max_life_x = max(datasets[g]['lifetime_centers'].max() for g in gammas)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Avalanche size (left panel, reference line only, inside) ---
    ax = axes[0]
    for i, gamma in enumerate(gammas):
        data = datasets[gamma]
        ax.loglog(data['size_centers'], data['size_probs'], 'o-',
                  markersize=3, alpha=0.8, color=colors[i])

    xs = np.logspace(0, np.log10(max_size_x), 100)
    last = datasets[gammas[-1]]
    y0 = last['size_probs'][0] if len(last['size_probs']) > 0 else 0.1
    ys = 10 * y0 * (xs / xs[0])**(-1.5)
    ax.loglog(xs, ys, 'k--', label=r'BP Theory $\tau=1.5$', linewidth=2)

    ax.set_xlabel('Avalanche Size $s$')
    ax.set_ylabel('Probability $P(s)$')
    ax.set_title('Avalanche Size Distribution ($\gamma$ Scale-Free)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # --- Avalanche lifetime (right panel, complete gamma legend outside) ---
    ax = axes[1]
    for i, gamma in enumerate(gammas):
        data = datasets[gamma]
        lbl = r'$\gamma = \infty$' if gamma >= 100 else rf'$\gamma = {gamma}$'
        ax.loglog(data['lifetime_centers'], data['lifetime_probs'], 'o-', label=lbl,
                  markersize=3, alpha=0.8, color=colors[i])

    xs = np.logspace(0, np.log10(max_life_x), 100)
    y0 = last['lifetime_probs'][0] if len(last['lifetime_probs']) > 0 else 0.1
    ys = 10 * y0 * (xs / xs[0])**(-2.0)
    ax.loglog(xs, ys, 'k--', label=r'BP Theory $z=2.0$', linewidth=2)

    ax.set_xlabel('Lifetime $t$')
    ax.set_ylabel('Probability $P(t)$')
    ax.set_title('Avalanche Lifetime Distribution ($\gamma$ Scale-Free)')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    fig.savefig(fig_dir / "task_15_part2.pdf", bbox_inches='tight')
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
