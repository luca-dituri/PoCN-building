"""Fit homogeneous and non-homogeneous graph kernels for the GridKit networks.

Run:  python code/task_47/run_kernel.py
Writes cleaned node/edge files and kernel artifacts under data/task_47/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "code"))

from common.utils import load_config, set_seed
from task_47.data_loading import prepare_network
from task_47 import kernel as K


def fit_network(network_key, config, raw_dir, out_dir, seed):
    data, diagnostics = prepare_network(network_key, config, out_dir, raw_dir)
    points, edges = data["points_xy_km"], data["edges"]
    N = points.shape[0]
    set_seed(seed)

    kcfg = config["kernel"]
    bin_edges, bin_stats = K.empirical_bin_stats(
        points, edges,
        n_bins=kcfg["n_bins"],
        bin_method=kcfg["bin_method"],
        n_sample_pairs=kcfg["n_sample_pairs"],
        seed=seed,
        r_log_floor=kcfg["r_log_floor"],
    )

    fits = {}
    for model in kcfg["models"]:
        fits[model] = K.fit_kernel(
            bin_stats, N, model=model,
            min_pairs=kcfg["min_pairs_per_bin"],
        )

    return {
        "network_key": network_key,
        "label": config["networks"][network_key]["label"],
        "diagnostics": diagnostics,
        "bin_edges": bin_edges,
        "bin_stats": bin_stats,
        "fits": fits,
    }


def main():
    config = load_config("task_47")
    raw_dir = BASE_DIR / "data" / "task_47" / "raw"
    out_dir = BASE_DIR / "data" / "task_47"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]

    results = {}
    for network_key in config["networks"]:
        print(f"Fitting kernels for {config['networks'][network_key]['label']} ...")
        results[network_key] = fit_network(network_key, config, raw_dir, out_dir, seed)

    # Persist kernel artifacts per network.
    for network_key, res in results.items():
        bs = res["bin_stats"]
        np.savez(
            out_dir / f"{network_key}_kernel.npz",
            r_mid=bs["r_mid"],
            pair_counts=bs["pair_counts"],
            link_counts=bs["link_counts"],
            empirical_p=bs["empirical_p"],
            diagnostics_N=res["diagnostics"]["N"],
            diagnostics_E=res["diagnostics"]["E"],
        )
        for model, fit in res["fits"].items():
            np.savez(
                out_dir / f"{network_key}_fit_{model}.npz",
                params_hat=fit["params_hat"],
                nll=fit["nll"], aic=fit["aic"], bic=fit["bic"],
                n_params=fit["n_params"], success=fit["success"],
                p_fit_bins=fit["p_fit_bins"],
            )

    # Model-selection summary CSV (NLL / AIC / BIC per network and model).
    rows = []
    for network_key, res in results.items():
        for model, fit in res["fits"].items():
            rows.append({
                "network": network_key,
                "model": model,
                "nll": fit["nll"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "n_params": fit["n_params"],
                "success": int(fit["success"]),
            })
    pd.DataFrame(rows).to_csv(out_dir / "kernel_model_comparison.csv", index=False)

    # Console summary.
    for network_key, res in results.items():
        d = res["diagnostics"]
        print(f"\n[{res['label']}] N={d['N']} E={d['E']} <k>={d['mean_degree']:.3f}")
        print(f"{'model':<22}{'scale (km)':>14}{'NLL':>16}{'AIC':>16}{'BIC':>16}")
        for model, fit in res["fits"].items():
            p = fit["params_hat"]
            key = p.get("ell", p.get("r0", np.nan))
            print(f"{model:<22}{key:>14.3f}{fit['nll']:>16.3e}{fit['aic']:>16.3e}{fit['bic']:>16.3e}")

    print("\nKernel fitting complete. Artifacts in data/task_47/")


if __name__ == "__main__":
    main()
