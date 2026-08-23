"""Infer Landau-Ginzburg (homogeneous + non-homogeneous) covariance models.

The covariance is the explicit quadratic operator Q = (kappa/h^2) L + r0 I
inverted numerically (G = T Q^{-1}) and fitted to the empirical graph kernel.
The non-homogeneous level is implemented over two coarse regions (dense /
sparse): the operator is fitted to each region's kernel and the two masses are
inverted for (r0, eps).

Run:  python code/task_47/run_lg.py
Requires the kernel artifacts produced by run_kernel.py.
Writes LG artifacts and comparison metrics under data/task_47/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "code"))

from common.utils import load_config
from task_47 import kernel as K
from task_47 import landau_ginzburg as LG


def load_cleaned(out_dir, network_key):
    nodes = pd.read_csv(out_dir / f"{network_key}_nodes.csv")
    edges = pd.read_csv(out_dir / f"{network_key}_edges.csv")
    points = np.column_stack([nodes["x_km"].values, nodes["y_km"].values])
    edge_arr = np.column_stack([edges["i"].values, edges["j"].values]).astype(np.int64)
    return points, edge_arr


def fit_region(points, edges, group, coords, h, kcfg, q_cfg, lg_cfg, seed,
               min_pairs):
    """Fit the homogeneous operator to a density-conditioned region kernel."""
    bin_edges = K.make_group_distance_bins(
        points, group, n_bins=kcfg["n_bins"], method=kcfg["bin_method"],
        n_sample_pairs=kcfg["n_sample_pairs"], seed=seed,
        r_log_floor=kcfg["r_log_floor"],
    )
    bs = K.density_split_bin_stats(points, edges, group, bin_edges)
    if np.sum((bs["pair_counts"] >= min_pairs) & (bs["empirical_p"] > 0)) < 5:
        return None
    return LG.fit_quadratic_covariance(
        coords, h, bin_edges, bs["r_mid"], bs["empirical_p"], bs["pair_counts"],
        kappa=lg_cfg["kappa"], temperature=lg_cfg["temperature"],
        min_pairs=min_pairs, r_min=q_cfg["r_min_factor"] * h,
        amp0=q_cfg["fit_amp0"], r00=q_cfg["fit_r00"], seed=seed,
    )


def main():
    config = load_config("task_47")
    out_dir = BASE_DIR / "data" / "task_47"
    lg_cfg = config["landau_ginzburg"]
    q_cfg = lg_cfg["quadratic"]
    kcfg = config["kernel"]
    rows = []

    for network_key in config["networks"]:
        label = config["networks"][network_key]["label"]
        print(f"Fitting Landau-Ginzburg operators for {label} ...")

        k = np.load(out_dir / f"{network_key}_kernel.npz")
        r_mid = k["r_mid"]
        empirical_p = k["empirical_p"]
        pair_counts = k["pair_counts"]

        points, edges = load_cleaned(out_dir, network_key)

        # Reproduce the exact bin edges used for the stored kernel (deterministic).
        bin_edges = K.make_distance_bins(
            points, n_bins=kcfg["n_bins"], method=kcfg["bin_method"],
            n_sample_pairs=kcfg["n_sample_pairs"], seed=config["seed"],
            r_log_floor=kcfg["r_log_floor"],
        )
        assert len(bin_edges) - 1 == len(r_mid), "bin edges / r_mid mismatch"

        coords, h = LG.make_grid_geometry(points, h_target=q_cfg["h_grid"])
        grid_n = int(round(np.sqrt(coords.shape[0])))
        r_min = q_cfg["r_min_factor"] * h
        print(f"  grid: n={grid_n}, h={h:.2f} km, r_min={r_min:.2f} km")

        # ---- homogeneous operator (single global mass)
        hom_margin = max(1, int(round(grid_n * q_cfg.get("source_margin_frac", 0.05))))
        hom = LG.fit_quadratic_covariance(
            coords, h, bin_edges, r_mid, empirical_p, pair_counts,
            kappa=lg_cfg["kappa"], temperature=lg_cfg["temperature"],
            min_pairs=kcfg["min_pairs_per_bin"], r_min=r_min,
            n_src=q_cfg["n_src"], seed=config["seed"],
            amp0=q_cfg["fit_amp0"], r00=q_cfg["fit_r00"],
            weighted=True, source_margin=hom_margin,
        )

        # ---- non-homogeneous operator over two coarse regions (dense / sparse)
        density = K.local_density(points, lg_cfg["density_radius"])
        m = LG.density_anomaly(density)
        dense = m >= 0.0

        group_min_pairs = max(kcfg["min_pairs_per_bin"] // 5, 50)
        hom_dense = fit_region(points, edges, dense, coords, h, kcfg, q_cfg,
                               lg_cfg, config["seed"], group_min_pairs)
        hom_sparse = fit_region(points, edges, ~dense, coords, h, kcfg, q_cfg,
                                lg_cfg, config["seed"], group_min_pairs)

        np.savez(
            out_dir / f"{network_key}_lg_homogeneous.npz",
            amplitude=hom["amplitude"], xi=hom["xi"], r0=hom["r0"],
            mse_log=hom["mse_log"], mse_lin=hom["mse_lin"], r2=hom["r2"],
            model_r=hom["model_r"], model_bins=hom["model_bins"],
            r_mid=r_mid, empirical_p=empirical_p, r_min=r_min,
        )

        # ---- perturbative extension: continuous mass field, O(eps) + O(u)
        m_grid = LG.grid_density_anomaly(points, coords, lg_cfg["density_radius"])
        dense_grid = m_grid >= 0.0

        Q0 = LG.build_precision(grid_n, h, lg_cfg["kappa"], hom["r0"])
        Q0_lu = LG.spla.splu(Q0.tocsc())

        groups = []
        rng = np.random.default_rng(config["seed"])
        for node_mask, grid_mask in ((dense, dense_grid), (~dense, ~dense_grid)):
            be = K.make_group_distance_bins(
                points, node_mask, n_bins=kcfg["n_bins"], method=kcfg["bin_method"],
                n_sample_pairs=kcfg["n_sample_pairs"], seed=config["seed"],
                r_log_floor=kcfg["r_log_floor"])
            bs = K.density_split_bin_stats(points, edges, node_mask, be)
            mask = ((bs["pair_counts"] >= group_min_pairs) & (bs["empirical_p"] > 0)
                    & np.isfinite(bs["empirical_p"]) & (bs["r_mid"] >= r_min))
            src_idx = np.where(grid_mask)[0]
            src = rng.choice(src_idx, size=min(q_cfg["n_src"], src_idx.size), replace=False)
            groups.append({"bin_edges": be, "r_mid": bs["r_mid"],
                           "empirical_p": bs["empirical_p"],
                           "pair_counts": bs["pair_counts"], "mask": mask,
                           "tgt_mask": grid_mask, "src": src})

        pert = LG.fit_perturbative_eps(
            Q0_lu, coords, m_grid, hom["r0"], lg_cfg["kappa"],
            lg_cfg["temperature"], groups, amp0=q_cfg["fit_amp0"], eps0=0.0,
            eps_bounds=tuple(q_cfg.get("eps_bounds", [-1.5, 1.5])))

        r_eff, phi2_0 = LG.one_loop_mass_shift(
            Q0_lu, grid_n, hom["r0"], lg_cfg["u"], lg_cfg["temperature"],
            n_diag=64, seed=config["seed"])
        xi_eff = float(np.sqrt(lg_cfg["kappa"] / max(r_eff, 1e-12)))

        src_all = rng.choice(coords.shape[0], size=min(q_cfg["n_src"], coords.shape[0]),
                             replace=False)
        g0_rad, dg_rad, g_rad = LG.radial_covariance_perturbative(
            Q0_lu, coords, bin_edges, lg_cfg["temperature"], m_grid,
            hom["r0"], pert["eps"], src_all)

        np.savez(
            out_dir / f"{network_key}_lg_perturbative.npz",
            r0=hom["r0"], eps=pert["eps"], amplitude=pert["amplitude"],
            success=pert["success"], loss=pert["loss"],
            phi2_0=phi2_0, u=lg_cfg["u"], r_eff=r_eff, xi_eff=xi_eff,
            m_grid=m_grid, dense_grid=dense_grid,
            dense_model_r=groups[0]["r_mid"][groups[0]["mask"]],
            dense_model_bins=pert["models"][0]["model_bins"],
            sparse_model_bins=pert["models"][1]["model_bins"],
            dense_emp=groups[0]["empirical_p"][groups[0]["mask"]],
            sparse_emp=groups[1]["empirical_p"][groups[1]["mask"]],
            g0_rad=g0_rad, dg_rad=dg_rad, g_rad=g_rad,
            r_mid=r_mid, grid_n=grid_n, h=h, r_min=r_min)

        pert_cols = {
            "pert_eps": pert["eps"],
            "pert_loss": pert["loss"],
            "phi2_0": phi2_0,
            "r_eff": r_eff,
            "xi_eff": xi_eff,
        }
        print(f"  perturbative: eps={pert['eps']:.3f} (success={pert['success']})  "
              f"phi2_0={phi2_0:.3f}  r_eff={r_eff:.3e}  xi_eff={xi_eff:.1f} km")

        if hom_dense is not None and hom_sparse is not None:
            nh = LG.infer_nonhomogeneous_coefficients(
                hom_dense["r0"], hom_sparse["r0"],
                m[dense].mean(), m[~dense].mean(), kappa=lg_cfg["kappa"],
            )
            np.savez(
                out_dir / f"{network_key}_lg_nonhomogeneous.npz",
                r0=nh["r0"], eps=nh["eps"],
                xi_dense=nh["xi_dense"], xi_sparse=nh["xi_sparse"],
                dense_r0=hom_dense["r0"], sparse_r0=hom_sparse["r0"],
                dense_amp=hom_dense["amplitude"], sparse_amp=hom_sparse["amplitude"],
                dense_r2=hom_dense["r2"], sparse_r2=hom_sparse["r2"],
                dense_model_r=hom_dense["model_r"], dense_model_bins=hom_dense["model_bins"],
                sparse_model_r=hom_sparse["model_r"], sparse_model_bins=hom_sparse["model_bins"],
                m_dense_mean=float(m[dense].mean()),
                m_sparse_mean=float(m[~dense].mean()),
                points=points, density=density, dense_mask=dense,
                density_radius=lg_cfg["density_radius"],
                grid_n=grid_n, h=h, r_min=r_min,
            )
            rows.append({
                "network": network_key,
                "xi_km": hom["xi"],
                "r0_homogeneous": hom["r0"],
                "lg_hom_mse_log": hom["mse_log"],
                "lg_hom_r2": hom["r2"],
                "xi_dense": nh["xi_dense"],
                "xi_sparse": nh["xi_sparse"],
                "nh_r0": nh["r0"],
                "nh_eps": nh["eps"],
                "dense_r2": hom_dense["r2"],
                "sparse_r2": hom_sparse["r2"],
                **pert_cols,
            })
            print(f"  homogeneous: r0={hom['r0']:.3e}  xi={hom['xi']:.2f} km  "
                  f"mse_log={hom['mse_log']:.4f}  r2={hom['r2']:.3f}")
            print(f"  non-homogeneous: r0={nh['r0']:.3e}  eps={nh['eps']:.3f}  "
                  f"xi_dense={nh['xi_dense']:.1f} km  xi_sparse={nh['xi_sparse']:.1f} km")
        else:
            rows.append({
                "network": network_key,
                "xi_km": hom["xi"],
                "r0_homogeneous": hom["r0"],
                "lg_hom_mse_log": hom["mse_log"],
                "lg_hom_r2": hom["r2"],
                **pert_cols,
            })
            print(f"  homogeneous: r0={hom['r0']:.3e}  xi={hom['xi']:.2f} km  "
                  f"mse_log={hom['mse_log']:.4f}  r2={hom['r2']:.3f}")
            print("  non-homogeneous: insufficient pairs in a density region; skipped")

    pd.DataFrame(rows).to_csv(out_dir / "lg_comparison.csv", index=False)
    print("\nLG inference complete. Artifacts in data/task_47/")


if __name__ == "__main__":
    main()
