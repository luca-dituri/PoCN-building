import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / 'code'))

from common.utils import load_config
from common.plot_utils import setup_plot_style, get_figure_dir, FONT_SIZE, LEGEND_SIZE

def plot_part1(data_dir, fig_dir):
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


def plot_fig4(data_dir, fig_dir):
    data = np.load(data_dir / "part3_fig4.npz")
    p = data['p_values']
    inset_p = data['inset_p_values']
    colors = ['#1f77b4', '#d62728', '#ffb000']

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_axes([0.16, 0.14, 0.78, 0.72])

    ax.errorbar(p, data['P_local'], yerr=data['P_local_err'], fmt='o-',
                color='#1f77b4', capsize=3, label=r'$T_{a,a}$')
    ax.errorbar(p, data['P_inflicted'], yerr=data['P_inflicted_err'], fmt='s-',
                color='#d62728', capsize=3, label=r'$T_{b,a}$')
    ax.errorbar(p, data['P_total'], yerr=data['P_total_err'], fmt='^-',
                color='#ffb000', capsize=3, label=r'$T_a$')

    ax.axvline(0.075, ls='--', color='k', alpha=0.6, lw=1.2, label=r'$p^{*}\approx0.075$')

    ax.set_xlabel('$p$')
    ax.set_ylabel(r'$P(T_a \geq 1000)$')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)

    # Downward translation of the data: expand the y-axis upward so the data
    # occupies the lower ~2/3 of the axes, leaving an empty band at the top.
    ymax = 0.0
    for arr, err in [(data['P_local'], data['P_local_err']),
                     (data['P_inflicted'], data['P_inflicted_err']),
                     (data['P_total'], data['P_total_err'])]:
        ymax = max(ymax, float(np.max(arr + err)))
    ax.set_ylim(0, 1.5 * ymax)

    # Inset: log-log rank-size (per-run average) of the 10^4 largest topplings in a
    axin = ax.inset_axes([0.55, 0.70, 0.36, 0.26])
    for j, pj in enumerate(inset_p):
        mean = data[f'top_ta_mean_{j}']
        ranks = np.arange(1, len(mean) + 1)
        axin.loglog(ranks, mean, 'o-', markersize=2.5, linewidth=1.0,
                    color=colors[j], label=f'$p=10^{{-{j+1}}}$')
    axin.set_xlabel('Rank', fontsize=LEGEND_SIZE)
    axin.set_ylabel(r'$T_a$', fontsize=LEGEND_SIZE)
    axin.grid(True, which="both", ls="--", alpha=0.3)
    axin.legend(fontsize=LEGEND_SIZE - 3, loc='lower right', framealpha=0.9)

    fig.savefig(fig_dir / "task_15_part3_fig4.pdf")
    plt.close(fig)

def plot_fig5(data_dir, fig_dir):
    data = np.load(data_dir / "part3_fig5.npz")
    inset_p = data['p_values']
    colors = ['#1f77b4', '#d62728', '#ffb000']

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_axes([0.15, 0.14, 0.79, 0.72])

    max_x = 1.0
    for j, pj in enumerate(inset_p):
        c = data[f'size_centers_{j}']
        pr = data[f'size_probs_{j}']
        ax.loglog(c, pr, 'o-', markersize=3, color=colors[j], label=f'$p=10^{{-{j+1}}}$')
        max_x = max(max_x, c.max())

    # Mean-field prediction s(t) ~ t^{-3/2}, anchored on the data (C ~ 0.5) and
    # raised by a factor 2 so the dashed line sits above the plotted data.
    xs = np.logspace(0, np.log10(max_x), 100)
    ys = 1.0 * xs ** (-1.5)
    ax.loglog(xs, ys, '--', color='g', lw=1.5, label=r'$s(t)\sim t^{-3/2}$')

    ax.set_xlabel('$t$')
    ax.set_ylabel('$s(t)$')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)

    # Downward translation of the data: expand the y-axis so the data occupies
    # the lower ~2/3 of the axes, leaving an empty band at the top clear of the
    # inset. On this log axis the top limit needs to be large (1e8) for the data
    # to stay below the inset area.
    ax.set_ylim(1e-13, 1e8)

    # Inset: log-log rank-size (per-run average) of the 10^4 largest global cascades
    axin = ax.inset_axes([0.55, 0.68, 0.36, 0.26])
    for j, pj in enumerate(inset_p):
        mean = data[f'top_total_mean_{j}']
        ranks = np.arange(1, len(mean) + 1)
        axin.loglog(ranks, mean, 'o-', markersize=2.5, linewidth=1.0,
                    color=colors[j])
    axin.set_xlabel('Rank', fontsize=LEGEND_SIZE)
    axin.set_ylabel(r'$t$', fontsize=LEGEND_SIZE)
    axin.grid(True, which="both", ls="--", alpha=0.3)

    fig.savefig(fig_dir / "task_15_part3_fig5.pdf")
    plt.close(fig)

def plot_fig6(data_dir, fig_dir):
    data = np.load(data_dir / "part3_fig6.npz")
    p = data['p_values']
    probs = data['window_probs']
    errs = data['window_errs']
    labels = [r'$I = [1 , 51]$', r'$I = [100 , 150]$',
              r'$I = [650 , 700 ]$', r'$I = [850 , 900]$']

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for idx, ax in enumerate(axes.flat):
        ax.errorbar(p, probs[idx], yerr=errs[idx], fmt='o-', capsize=3)
        ax.set_title(f'({chr(65 + idx)})  {labels[idx]}')
        ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.supxlabel('$p$', fontsize=FONT_SIZE)
    # Row titles and the shared probability label sit in the left margin, to the
    # left of the numeric y-tick labels (margin widened via subplots_adjust).
    fig.text(0.05, 0.72, 'Small cascades', rotation=90, va='center',
             fontsize=16, fontweight='bold', color='#1f77b4')
    fig.text(0.05, 0.28, 'Large cascades', rotation=90, va='center',
             fontsize=16, fontweight='bold', color='#d62728')
    fig.text(0.012, 0.5, r'$P(T_a \in I)$', rotation=90, va='center', fontsize=15)

    fig.tight_layout()
    fig.subplots_adjust(left=0.17)
    fig.savefig(fig_dir / "task_15_part3_fig6.pdf")
    plt.close(fig)


def main():
    config = load_config("task_15")
    data_dir = base_dir / "data" / "task_15"
    fig_dir = get_figure_dir("task_15")

    setup_plot_style()

    print("Plotting Part 1...")
    plot_part1(data_dir, fig_dir)

    print("Plotting Part 2...")
    plot_part2(config, data_dir, fig_dir)

    print("Plotting Part 3 (Fig 4)...")
    plot_fig4(data_dir, fig_dir)
    print("Plotting Part 3 (Fig 5)...")
    plot_fig5(data_dir, fig_dir)
    print("Plotting Part 3 (Fig 6)...")
    plot_fig6(data_dir, fig_dir)

    print("Done! Plots saved in latex/figures/task_15/")

if __name__ == "__main__":
    main()
