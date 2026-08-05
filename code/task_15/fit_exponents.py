import sys
from pathlib import Path
import argparse
import numpy as np

base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / 'code'))

from common.utils import load_config

TAU_RANDOM = 1.5
Z_RANDOM = 2.0

# Candidate lower cutoffs for the plateau detector (bins whose center >= value).
CAND_SMIN = [5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]
CAND_TMIN = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
SW_WINDOW = 5      # sliding window (bins) for local-slope stability
SW_TOL = 0.25      # max deviation from median local slope to count as "flat"
MIN_RUN_BINS = 6   # minimum number of bins in an accepted plateau run
MIN_LOGSPAN = 0.8  # minimum log10(hi/lo) span of an accepted plateau


def theory_tau(gamma):
    if 2.0 < gamma < 3.0:
        return gamma / (gamma - 1.0)
    return TAU_RANDOM


def theory_z(gamma):
    if 2.0 < gamma < 3.0:
        return (gamma - 1.0) / (gamma - 2.0)
    return Z_RANDOM


def fit_powerlaw(centers, probs, smin=None, smax=None, n_tail=0):
    """Fit log10(prob) vs log10(center) with weighted least squares.

    probs are densities from monotonic log-binning, so the count in a bin is
    approximately proportional to probs * center (bin width ~ center). Returns
    (exponent, se, r2, n_points, xmin_used, xmax_used).
    """
    centers = np.asarray(centers, dtype=float)
    probs = np.asarray(probs, dtype=float)

    if n_tail > 0 and len(centers) > n_tail + 2:
        centers = centers[:-n_tail]
        probs = probs[:-n_tail]

    if smin is not None:
        mask = centers >= smin
        centers = centers[mask]
        probs = probs[mask]

    if smax is not None:
        mask = centers <= smax
        centers = centers[mask]
        probs = probs[mask]

    if len(centers) < 3:
        return np.nan, np.nan, np.nan, 0, np.nan, np.nan

    x = np.log10(centers)
    y = np.log10(probs)
    w = probs * centers  # ~ count per bin
    w = w / w.sum()

    A = np.vstack([x, np.ones_like(x)]).T
    Aw = A * np.sqrt(w)[:, None]
    bw = y * np.sqrt(w)

    coeff, _, rank, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    slope = coeff[0]
    y_pred = A @ coeff

    n = len(x)
    if n > 2 and rank == 2:
        resid = y - y_pred
        s2 = np.sum(w * resid ** 2) / (n - 2)
        cov = s2 * np.linalg.inv(Aw.T @ Aw)
        se = np.sqrt(max(cov[0, 0], 0.0))
    else:
        se = np.nan

    ss_res = np.sum(w * (y - y_pred) ** 2)
    ss_tot = np.sum(w * (y - np.sum(w * y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # prob ~ center^slope  =>  exponent = -slope
    return -slope, se, r2, n, centers.min(), centers.max()


def local_slopes(centers, probs, window=SW_WINDOW):
    """Sliding-window log-log slopes: one (center, -slope) per window position."""
    x = np.log10(centers)
    y = np.log10(probs)
    out = []
    for i in range(len(x) - window + 1):
        m, _ = np.polyfit(x[i:i + window], y[i:i + window], 1)
        out.append((centers[i + window // 2], -m))
    return np.array(out)


def find_plateau(centers, probs, cand, n_tail=5, min_run_bins=MIN_RUN_BINS,
                 min_logspan=MIN_LOGSPAN):
    """Pick the flattest scaling window among candidate lower cutoffs.

    A window of fixed span (min_logspan decades) is slid across the data; for
    every window whose left edge is at/after a candidate lower cutoff we
    compute the local-slope spread (sliding-window slopes from `local_slopes`).
    The window with the smallest slope variance is the plateau: that is where
    the log-log curve is straightest. Among candidates the tightest window
    wins, tie-broken by the largest lower cutoff (skips the small-s roll-off).
    Returns (smin_used, smax_used) or None if no plateau is found.
    """
    centers = np.asarray(centers, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if n_tail > 0 and len(centers) > n_tail + 2:
        centers = centers[:-n_tail]
        probs = probs[:-n_tail]
    if len(centers) < min_run_bins + SW_WINDOW:
        return None

    slopes = local_slopes(centers, probs)
    s_c = slopes[:, 0]
    s_e = slopes[:, 1]

    # scan windows with left edge at/after each candidate cutoff
    best = None  # (run_std, -smin, lo, hi)
    for smin in cand:
        for i in range(len(s_c)):
            if s_c[i] < smin:
                continue
            lo = s_c[i]
            hi_goal = lo * 10 ** min_logspan
            j = i
            while j + 1 < len(s_c) and s_c[j + 1] <= hi_goal:
                j += 1
            if j - i + 1 < min_run_bins:
                continue
            run_std = np.std(s_e[i:j + 1])
            hi = s_c[j]
            if best is None or (run_std, -lo) < (best[0], best[1]):
                best = (run_std, -lo, lo, hi)

    if best is None:
        return None
    return best[2], best[3]


def fit_dataset(npz_path, name, gamma=None, smin=None, tmin=None, n_tail=5, use_plateau=True):
    data = np.load(npz_path)

    if use_plateau and smin is None:
        p = find_plateau(data['size_centers'], data['size_probs'], CAND_SMIN, n_tail)
        smin = p[0] if p else None
        smax_size = p[1] if p else None
    else:
        smax_size = None

    if use_plateau and tmin is None:
        p = find_plateau(data['lifetime_centers'], data['lifetime_probs'], CAND_TMIN, n_tail)
        tmin = p[0] if p else None
        smax_life = p[1] if p else None
    else:
        smax_life = None

    tau, tau_se, tau_r2, n_s, smin_a, smax_a = fit_powerlaw(
        data['size_centers'], data['size_probs'], smin=smin, smax=smax_size, n_tail=n_tail)
    z, z_se, z_r2, n_l, lmin_a, lmax_a = fit_powerlaw(
        data['lifetime_centers'], data['lifetime_probs'], smin=tmin, smax=smax_life, n_tail=n_tail)

    ttau = theory_tau(gamma) if gamma is not None else TAU_RANDOM
    tz = theory_z(gamma) if gamma is not None else Z_RANDOM

    return {
        'name': name,
        'tau': tau, 'tau_se': tau_se, 'tau_r2': tau_r2,
        'z': z, 'z_se': z_se, 'z_r2': z_r2,
        'tau_theory': ttau, 'z_theory': tz,
        'n_size': n_s, 'n_life': n_l,
        's_range': (smin_a, smax_a), 't_range': (lmin_a, lmax_a),
    }


def main():
    ap = argparse.ArgumentParser(description='Fit power-law exponents from binned data.')
    ap.add_argument('--smin', type=float, default=None,
                    help='fixed lower cutoff on avalanche size centers (default: plateau detector)')
    ap.add_argument('--tmin', type=float, default=None,
                    help='fixed lower cutoff on lifetime centers (default: plateau detector)')
    ap.add_argument('--n-tail', type=int, default=5,
                    help='number of trailing noisy bins to exclude')
    ap.add_argument('--no-plateau', action='store_true',
                    help='disable the plateau detector (requires --smin/--tmin)')
    ap.add_argument('--sweep', action='store_true',
                    help='print a tau/z vs fitting-window sensitivity matrix')
    ap.add_argument('--no-save', action='store_true', help='do not write fits.csv')
    args = ap.parse_args()

    config = load_config("task_15")
    data_dir = base_dir / "data" / "task_15"

    results = []

    # Part 1: Gaussian & Uniform (random graph theory: tau=1.5, z=2)
    for name, fname in [("Gaussian", "part1_gaussian.npz"), ("Uniform", "part1_uniform.npz")]:
        results.append(fit_dataset(data_dir / fname, name, gamma=None,
                                   smin=args.smin, tmin=args.tmin, n_tail=args.n_tail,
                                   use_plateau=not args.no_plateau))

    # Part 2: Scale-free (gamma)
    for gamma in config['part2']['gammas']:
        lbl = r'$\gamma = \infty$' if gamma >= 100 else rf'$\gamma = {gamma}$'
        fname = f"part2_gamma_{gamma}.npz"
        results.append(fit_dataset(data_dir / fname, lbl, gamma=float(gamma),
                                   smin=args.smin, tmin=args.tmin, n_tail=args.n_tail,
                                   use_plateau=not args.no_plateau))

    if args.sweep:
        print_sweep(config, data_dir, args.n_tail)

    # Print summary
    print(f"{'Network':<16}{'tau_theory':>11}{'tau_fit':>9}{'tau_se':>8}{'z_theory':>9}{'z_fit':>9}{'z_se':>8}  s-range       t-range")
    print("-" * 104)
    for r in results:
        def fmt(v):
            return f"{v:.3f}" if np.isfinite(v) else "  -  "
        print(f"{r['name']:<16}{fmt(r['tau_theory']):>11}{fmt(r['tau']):>9}{fmt(r['tau_se']):>8}"
              f"{fmt(r['z_theory']):>9}{fmt(r['z']):>9}{fmt(r['z_se']):>8}"
              f"  [{r['s_range'][0]:.0f},{r['s_range'][1]:.0e}] [{r['t_range'][0]:.0f},{r['t_range'][1]:.0e}]")

    # LaTeX table snippet
    print("\n--- LaTeX table ---")
    print(r"\begin{table}[htbp]")
    print(r"    \centering")
    print(r"    \begin{tabular}{lc|cc|cc}")
    print(r"        \toprule")
    print(r"        \textbf{Network} && \multicolumn{2}{c|}{$\tau$} & \multicolumn{2}{c}{$z$} \\")
    print(r"        & & theory & fit & theory & fit \\")
    print(r"        \midrule")
    for r in results:
        if np.isfinite(r['tau']):
            tau_cell = rf"{r['tau']:.2f} $\pm$ {r['tau_se']:.2f}"
        else:
            tau_cell = "--"
        if np.isfinite(r['z']):
            z_cell = rf"{r['z']:.2f} $\pm$ {r['z_se']:.2f}"
        else:
            z_cell = "--"
        name_tex = r['name'] if not r['name'].startswith('$') else r['name']
        print(f"        {name_tex} && {r['tau_theory']:.2f} & {tau_cell}"
              f" & {r['z_theory']:.2f} & {z_cell} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \caption{Fitted avalanche-size ($\tau$) and lifetime ($z$) exponents vs multiplicative branching-process predictions."
          r" Fit over the plateau of the log-binned distributions (weighted least squares, per-row ranges in Table~\ref{tab:fit_ranges}).}")
    print(r"    \label{tab:fitted_exponents}")
    print(r"\end{table}")

    print("\n--- Fitting ranges ---")
    print(r"\begin{table}[htbp]")
    print(r"    \centering")
    print(r"    \begin{tabular}{lc|cc|cc}")
    print(r"        \toprule")
    print(r"        \textbf{Network} && \multicolumn{2}{c|}{$s$} & \multicolumn{2}{c}{$t$} \\")
    print(r"        & & min & max & min & max \\")
    print(r"        \midrule")
    for r in results:
        name_tex = r['name'] if not r['name'].startswith('$') else r['name']
        s_min = r['s_range'][0] if np.isfinite(r['s_range'][0]) else 0.0
        s_max = r['s_range'][1] if np.isfinite(r['s_range'][1]) else 0.0
        t_min = r['t_range'][0] if np.isfinite(r['t_range'][0]) else 0.0
        t_max = r['t_range'][1] if np.isfinite(r['t_range'][1]) else 0.0
        print(f"        {name_tex} && {s_min:.0f} & {s_max:.0e} & {t_min:.0f} & {t_max:.0e} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \caption{Scaling ranges used for the exponent fits of Table~\ref{tab:fitted_exponents}.}")
    print(r"    \label{tab:fit_ranges}")
    print(r"\end{table}")

    if not args.no_save:
        import csv
        out = data_dir / "fits.csv"
        with open(out, "w", newline="") as f:
            fieldnames = ['network', 'tau_theory', 'tau_fit', 'tau_se', 'tau_r2',
                          'z_theory', 'z_fit', 'z_se', 'z_r2', 'n_size', 'n_life',
                          's_min', 's_max', 't_min', 't_max']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'network': r['name'], 'tau_theory': r['tau_theory'],
                    'tau_fit': r['tau'], 'tau_se': r['tau_se'], 'tau_r2': r['tau_r2'],
                    'z_theory': r['z_theory'], 'z_fit': r['z'], 'z_se': r['z_se'],
                    'z_r2': r['z_r2'], 'n_size': r['n_size'], 'n_life': r['n_life'],
                    's_min': r['s_range'][0], 's_max': r['s_range'][1],
                    't_min': r['t_range'][0], 't_max': r['t_range'][1],
                })
        print(f"\nfits.csv written to {out}")


def print_sweep(config, data_dir, n_tail):
    """Print a tau/z vs window sensitivity matrix for transparency."""
    windows_s = [(None, None)] + [(lo, hi) for lo in [5, 10, 20, 30, 50]
                                  for hi in [200, 500, 1000]]
    windows_t = [(None, None)] + [(lo, hi) for lo in [4, 6, 10]
                                  for hi in [40, 80, 150]]

    for gamma in [None] + [float(g) for g in config['part2']['gammas']]:
        if gamma is None:
            fname = "part1_gaussian.npz"
            name = "Gaussian"
            g = None
        else:
            fname = f"part2_gamma_{gamma}.npz"
            name = rf"$\gamma = {gamma}$"
            g = gamma
        data = np.load(data_dir / fname)
        print(f"\n--- {name} (n_tail={n_tail}) ---")
        print("    tau:  [default/plateau]  then window -> exponent")
        tau_def = fit_powerlaw(data['size_centers'], data['size_probs'],
                               smin=8.0, n_tail=n_tail)[0]
        print(f"    smin=8 fixed: {tau_def:.3f}")
        for lo, hi in windows_s:
            if lo is None and hi is None:
                continue
            t = fit_powerlaw(data['size_centers'], data['size_probs'],
                             smin=lo, smax=hi, n_tail=n_tail)[0]
            print(f"    s in [{lo:2d},{hi:>4}] -> tau = {t:.3f}")
        z_def = fit_powerlaw(data['lifetime_centers'], data['lifetime_probs'],
                             smin=4.0, n_tail=n_tail)[0]
        print(f"    tmin=4 fixed: z = {z_def:.3f}")
        for lo, hi in windows_t:
            if lo is None and hi is None:
                continue
            z = fit_powerlaw(data['lifetime_centers'], data['lifetime_probs'],
                             smin=lo, smax=hi, n_tail=n_tail)[0]
            print(f"    t in [{lo:2d},{hi:>3}] -> z  = {z:.3f}")


if __name__ == "__main__":
    main()
