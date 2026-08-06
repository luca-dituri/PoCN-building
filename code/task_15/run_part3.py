import sys
from pathlib import Path
import numpy as np
from multiprocessing import Pool
import time

base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / 'code'))

from common.utils import load_config, set_seed
from task_15.network_generation import generate_random_regular_interconnected
from task_15.sandpile import simulate_sandpile_split

BURN_IN_FRAC = 0.2
NUM_BINS = 48
MAX_TOTAL = 2 ** 17

def build_total_edges(max_val, num_bins):
    edges = np.geomspace(1.0, float(max_val), num_bins + 1)
    return edges

def sanitize(p):
    return f"{p:g}".replace('.', '_')

def top_k_sorted(values, k):
    if len(values) <= k:
        return np.sort(values)
    return np.sort(np.partition(values, len(values) - k)[-k:])

def process_task(args):
    seed, p, config = args
    set_seed(seed)

    N = config['part3']['N']
    degree = config['part3']['degree']
    f = config['part3']['f']
    grains = config['part3']['grains']
    big_frac = config['part3']['big_threshold_frac']
    top_k = config['part3']['top_k']
    inset_p_values = config['part3']['inset_p_values']
    windows = np.asarray(config['part3']['fig6_windows'], dtype=np.int64)

    total_drops = int(grains / (1.0 - BURN_IN_FRAC))
    G, network_ids = generate_random_regular_interconnected(N, degree, p)
    origins, ta, tb = simulate_sandpile_split(
        G, network_ids, num_drops=total_drops, f=f, burn_in_frac=BURN_IN_FRAC)

    n_drops = len(ta)
    threshold = int(big_frac * N)
    big = ta > threshold

    big_local = int(np.sum(big & (origins == 0)))
    big_inflicted = int(np.sum(big & (origins == 1)))
    big_total = int(np.sum(big))

    totals = ta + tb
    window_counts = np.array(
        [int(np.sum((ta >= lo) & (ta <= hi))) for lo, hi in windows], dtype=np.int64)

    result = {
        'n_drops': n_drops,
        'big_local': big_local,
        'big_inflicted': big_inflicted,
        'big_total': big_total,
        'window_counts': window_counts,
    }

    if p in inset_p_values:
        edges = build_total_edges(MAX_TOTAL, NUM_BINS)
        counts, _ = np.histogram(totals, bins=edges)
        result['top_ta'] = top_k_sorted(ta, top_k)
        result['top_total'] = top_k_sorted(totals, top_k)
        result['hist_counts'] = counts

    return result

def main():
    config = load_config("task_15")
    base_seed = config['seed']
    p_values = config['part3']['p_values']
    inset_p_values = config['part3']['inset_p_values']
    realizations = config['part3']['realizations']

    data_dir = base_dir / "data" / "task_15"
    data_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(base_seed + 100000 * pi + r, p, config)
             for pi, p in enumerate(p_values)
             for r in range(realizations)]

    print(f"Running Part 3: {len(p_values)} p-values x {realizations} realizations = {len(tasks)} tasks")
    t0 = time.time()

    with Pool() as pool:
        results = pool.map(process_task, tasks)
    print(f"Simulation done in {time.time() - t0:.2f}s")

    n_p = len(p_values)
    P_local = np.zeros(n_p)
    P_local_std = np.zeros(n_p)
    P_inflicted = np.zeros(n_p)
    P_inflicted_std = np.zeros(n_p)
    P_total = np.zeros(n_p)
    P_total_std = np.zeros(n_p)

    n_windows = len(config['part3']['fig6_windows'])
    win_means = np.zeros((n_windows, n_p))
    win_stds = np.zeros((n_windows, n_p))

    top_ta_mean_by_p = {}
    top_ta_std_by_p = {}
    top_total_mean_by_p = {}
    top_total_std_by_p = {}
    hist_counts_by_p = {}

    for pi, p in enumerate(p_values):
        for r in range(realizations):
            res = results[pi * realizations + r]
            nd = res['n_drops']
            P_local[pi] += res['big_local'] / nd
            P_inflicted[pi] += res['big_inflicted'] / nd
            P_total[pi] += res['big_total'] / nd
            win_means[:, pi] += res['window_counts'] / nd

        P_local[pi] /= realizations
        P_inflicted[pi] /= realizations
        P_total[pi] /= realizations
        win_means[:, pi] /= realizations

        per_real_local = np.array([results[pi * realizations + r]['big_local'] / results[pi * realizations + r]['n_drops']
                                   for r in range(realizations)])
        per_real_inflicted = np.array([results[pi * realizations + r]['big_inflicted'] / results[pi * realizations + r]['n_drops']
                                       for r in range(realizations)])
        per_real_total = np.array([results[pi * realizations + r]['big_total'] / results[pi * realizations + r]['n_drops']
                                   for r in range(realizations)])
        P_local_std[pi] = per_real_local.std(ddof=1) / np.sqrt(realizations)
        P_inflicted_std[pi] = per_real_inflicted.std(ddof=1) / np.sqrt(realizations)
        P_total_std[pi] = per_real_total.std(ddof=1) / np.sqrt(realizations)

        per_real_win = np.array([[results[pi * realizations + r]['window_counts'][w] / results[pi * realizations + r]['n_drops']
                                  for r in range(realizations)] for w in range(n_windows)])
        win_stds[:, pi] = per_real_win.std(axis=1, ddof=1) / np.sqrt(realizations)

        if p in inset_p_values:
            per_run_ta = np.stack(
                [results[pi * realizations + r]['top_ta'][::-1] for r in range(realizations)])
            per_run_total = np.stack(
                [results[pi * realizations + r]['top_total'][::-1] for r in range(realizations)])
            top_ta_mean_by_p[p] = per_run_ta.mean(axis=0)
            top_ta_std_by_p[p] = per_run_ta.std(axis=0, ddof=1)
            top_total_mean_by_p[p] = per_run_total.mean(axis=0)
            top_total_std_by_p[p] = per_run_total.std(axis=0, ddof=1)
            total_counts = np.sum([results[pi * realizations + r]['hist_counts'] for r in range(realizations)], axis=0)
            hist_counts_by_p[p] = total_counts

    p_arr = np.array(p_values, dtype=float)
    np.savez(data_dir / "part3_fig4.npz",
             p_values=p_arr,
             P_local=P_local, P_local_err=P_local_std,
             P_inflicted=P_inflicted, P_inflicted_err=P_inflicted_std,
             P_total=P_total, P_total_err=P_total_std,
             inset_p_values=np.array(inset_p_values, dtype=float),
             top_ta_mean_0=top_ta_mean_by_p[inset_p_values[0]],
             top_ta_std_0=top_ta_std_by_p[inset_p_values[0]],
             top_ta_mean_1=top_ta_mean_by_p[inset_p_values[1]],
             top_ta_std_1=top_ta_std_by_p[inset_p_values[1]],
             top_ta_mean_2=top_ta_mean_by_p[inset_p_values[2]],
             top_ta_std_2=top_ta_std_by_p[inset_p_values[2]])

    edges = build_total_edges(MAX_TOTAL, NUM_BINS)
    bin_widths = np.diff(edges)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    total_kept = config['part3']['grains'] * realizations

    fig5_arrays = {}
    for j, p in enumerate(inset_p_values):
        counts = hist_counts_by_p[p]
        probs = counts / (total_kept * bin_widths)
        valid = counts > 0
        fig5_arrays[f"size_centers_{j}"] = bin_centers[valid]
        fig5_arrays[f"size_probs_{j}"] = probs[valid]
        fig5_arrays[f"top_total_mean_{j}"] = top_total_mean_by_p[p]
        fig5_arrays[f"top_total_std_{j}"] = top_total_std_by_p[p]

    np.savez(data_dir / "part3_fig5.npz",
             p_values=np.array(inset_p_values, dtype=float),
             **fig5_arrays)

    np.savez(data_dir / "part3_fig6.npz",
             p_values=p_arr,
             windows=np.asarray(config['part3']['fig6_windows'], dtype=np.int64),
             window_probs=win_means,
             window_errs=win_stds)

    print("Part 3 complete. Data saved to data/task_15/part3_{fig4,fig5,fig6}.npz")
    for pi, p in enumerate(p_values):
        print(f"  p={p:g}: P_local={P_local[pi]:.3e}  P_inflicted={P_inflicted[pi]:.3e}  P_total={P_total[pi]:.3e}")

if __name__ == "__main__":
    main()
