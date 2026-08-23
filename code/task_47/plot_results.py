"""Plot empirical kernels, parametric fits, and Landau-Ginzburg covariance.

Run:  python code/task_47/plot_results.py
Writes figures to latex/figures/task_47/.

Homogeneous panel: two stacked axes (log-log fit on top, standardized residuals
below) so the overlapping candidate kernels and the LG operator can be told
apart; the residual panel shows (data - model) normalized by the per-bin
binomial standard error, with a zero line and a +/-3 sigma band.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "code"))

from common.utils import load_config
from common.plot_utils import setup_plot_style, get_figure_dir

# colour / label for each parametric kernel (kept consistent across panels)
_MODEL_STYLES = {
    "exponential": ("#1f77b4", "exponential"),
    "gaussian": ("#ff7f0e", "gaussian"),
    "powerlaw_cutoff": ("#2ca02c", "power-law cutoff"),
    "stretched_exponential": ("#9467bd", "stretched exp."),
    "exp_power": ("#d62728", "exp. $\\times$ pow."),
}
_LG_COLOR = "tab:blue"


def _empirical_mask(bs):
    return (bs["pair_counts"] > 0) & (bs["link_counts"] > 0) & np.isfinite(bs["empirical_p"])


def _residuals(r, empirical, model, pair_counts):
    """Pearson (standardized) residuals (empirical - model) / sigma at bins where
    both are defined, with sigma = sqrt(model (1 - model) / pair_counts) the
    binomial standard error of the probability estimate."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.sqrt(model * (1.0 - model) / pair_counts)
        resid = (empirical - model) / sigma
    ok = (np.isfinite(resid) & (model > 0) & (empirical > 0)
          & (pair_counts > 0))
    return r[ok], resid[ok]


def _best_model(out_dir, network_key):
    comp = pd.read_csv(out_dir / "kernel_model_comparison.csv")
    sub = comp[comp["network"] == network_key]
    return sub.sort_values("aic")["model"].iloc[0]


def _group_stats(points, edges, group, config):
    from task_47 import kernel as K
    kcfg = config["kernel"]
    bin_edges = K.make_group_distance_bins(
        points, group, n_bins=kcfg["n_bins"], method=kcfg["bin_method"],
        n_sample_pairs=kcfg["n_sample_pairs"], seed=config["seed"],
        r_log_floor=kcfg["r_log_floor"],
    )
    return K.density_split_bin_stats(points, edges, group, bin_edges)


def main():
    config = load_config("task_47")
    out_dir = BASE_DIR / "data" / "task_47"
    fig_dir = get_figure_dir("task_47")
    setup_plot_style()

    for network_key in config["networks"]:
        label = config["networks"][network_key]["label"]

        k = np.load(out_dir / f"{network_key}_kernel.npz")
        r = k["r_mid"]
        empirical_p = k["empirical_p"]
        m = _empirical_mask(k)

        x_lo = max(0.3, 0.5 * float(r[m].min()))
        x_hi = 1.15 * float(r[m].max())

        hom = np.load(out_dir / f"{network_key}_lg_homogeneous.npz")
        best = _best_model(out_dir, network_key)

        # --- two-row layout: homogeneous (top log-log + bottom ratio) | non-homogeneous
        fig = plt.figure(figsize=(13, 7.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.10, wspace=0.32)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_ratio = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_nh = fig.add_subplot(gs[:, 1])

        # ---- top-left: log-log data + best parametric model + LG operator
        ax_main.scatter(r[m], empirical_p[m], s=18, color="black",
                        label="empirical $p(r)$")
        ax_main.plot(hom["model_r"], hom["model_bins"], lw=2, ls="--",
                     color=_LG_COLOR,
                     label=r"LG, $\xi=%.0f$ km" % hom["xi"])
        bc, bdisp = _MODEL_STYLES[best]
        fbest = np.load(out_dir / f"{network_key}_fit_{best}.npz")
        ax_main.plot(r, fbest["p_fit_bins"], lw=1.8, color=bc,
                     label=f"{bdisp} (best)")

        ax_main.set_xscale("log"); ax_main.set_yscale("log")
        ax_main.set_ylabel("connection probability $p(r)$")
        ax_main.set_title(f"{label} — homogeneous")
        ax_main.legend(fontsize=9, loc="lower left")
        ax_main.set_xlim(x_lo, x_hi)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # ---- bottom-left: standardized residuals ((data - model) / sigma)
        ax_ratio.axhline(0.0, color="gray", lw=1, ls=":")
        ax_ratio.axhline(3.0, color="gray", lw=0.6, ls=":", alpha=0.6)
        ax_ratio.axhline(-3.0, color="gray", lw=0.6, ls=":", alpha=0.6)
        exclude = set(config["kernel"].get("residual_exclude", []))
        for model in config["kernel"]["models"]:
            if model in exclude:
                continue
            c, disp = _MODEL_STYLES[model]
            f = np.load(out_dir / f"{network_key}_fit_{model}.npz")
            rr, res = _residuals(r, empirical_p, f["p_fit_bins"], k["pair_counts"])
            ax_ratio.plot(rr, res, lw=1.2, color=c, label=disp)

        # LG operator residuals (only defined for r >= lattice spacing h)
        lg_model = np.full_like(r, np.nan)
        ok = r >= hom["model_r"].min()
        lg_model[ok] = np.exp(np.interp(np.log(r[ok]),
                                        np.log(hom["model_r"]),
                                        np.log(hom["model_bins"])))
        rr, res = _residuals(r, empirical_p, lg_model, k["pair_counts"])
        ax_ratio.plot(rr, res, lw=1.4, ls="--", color=_LG_COLOR, label="LG")

        ax_ratio.set_xscale("log")
        ax_ratio.set_xlabel("distance $r$ [km]")
        ax_ratio.set_ylabel("residual $(p - \\mathrm{model})/\\sigma$")
        ax_ratio.set_xlim(x_lo, x_hi)
        ax_ratio.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)

        # ---- right: non-homogeneous (dense vs sparse) + operator fit
        try:
            nh = np.load(out_dir / f"{network_key}_lg_nonhomogeneous.npz")
        except FileNotFoundError:
            nh = None

        if nh is not None:
            dense_mask = nh["dense_mask"].astype(bool)
            edges = np.column_stack([
                pd.read_csv(out_dir / f"{network_key}_edges.csv")["i"].values,
                pd.read_csv(out_dir / f"{network_key}_edges.csv")["j"].values,
            ]).astype(np.int64)

            for group, name, color in [(dense_mask, "dense", "tab:red"),
                                       (~dense_mask, "sparse", "tab:blue")]:
                bs = _group_stats(nh["points"], edges, group, config)
                gm = _empirical_mask(bs)
                ax_nh.scatter(bs["r_mid"][gm], bs["empirical_p"][gm], s=18,
                              color=color, alpha=1, label=f"{name} $p(r)$")
                key = "dense" if name == "dense" else "sparse"
                ax_nh.plot(nh[f"{key}_model_r"], nh[f"{key}_model_bins"], lw=1.5,
                           color=color,
                           label=rf"{name}, $\xi={nh['xi_'+key]:.0f}$ km")

        ax_nh.set_xscale("log"); ax_nh.set_yscale("log")
        ax_nh.set_xlabel("distance $r$ [km]")
        ax_nh.set_ylabel("connection probability $p(r)$")
        ax_nh.set_title(f"{label} — non-homogeneous")
        ax_nh.legend(fontsize=9, loc="lower left")
        ax_nh.set_xlim(x_lo, x_hi)

        if nh is not None:
            fig.canvas.draw()
            lb = ax_nh.get_legend().get_window_extent().transformed(ax_nh.transAxes.inverted())
            ax_nh.text(lb.x0, lb.y1 + 0.02,
                       rf"$r_0={nh['r0']:.2e}$, $\varepsilon={nh['eps']:.2f}$ ",
                       transform=ax_nh.transAxes, fontsize=11, ha="left", va="bottom",
                       bbox=dict(fc="white", ec="gray", alpha=0.85))

        fig.savefig(fig_dir / f"task_47_kernel_lg_{network_key}.pdf",
                    bbox_inches="tight")
        plt.close(fig)

        # --- density map
        if nh is not None:
            points = nh["points"]; density = nh["density"]
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(points[:, 0], points[:, 1], c=density, s=4,
                            cmap="viridis", alpha=0.7)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("local density (nodes within %.0f km)"
                         % float(nh["density_radius"]))
            ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
            ax.set_title(f"{label} — spatial density")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(fig_dir / f"task_47_density_map_{network_key}.pdf",
                        bbox_inches="tight")
            plt.close(fig)

    print("Plots written to", fig_dir)


if __name__ == "__main__":
    main()
