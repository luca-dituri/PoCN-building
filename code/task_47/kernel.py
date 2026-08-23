"""Empirical graph kernels for embedded networks.

Estimates the radial connection probability p(r) = P(connected | distance r)
by binning node pairs by distance, and fits parametric kernels by binomial
(binned) log-likelihood. Also provides a grid-cell (non-homogeneous) kernel
and cell density used by the Landau-Ginzburg comparison.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize

from task_47.data_loading import BASE_DIR


# ---------------------------------------------------------------- binning

@njit
def _euclidean_distance_2d(points, i, j):
    dx = points[i, 0] - points[j, 0]
    dy = points[i, 1] - points[j, 1]
    return np.sqrt(dx * dx + dy * dy)


@njit
def _find_bin(d, bin_edges):
    '''
    Given d distance between two points, finds which bin d is in
    '''
    n_bins = bin_edges.shape[0] - 1
    if d < bin_edges[0] or d >= bin_edges[-1]:
        return -1
    lo, hi = 0, n_bins
    while lo < hi:
        mid = (lo + hi) // 2
        if bin_edges[mid + 1] <= d:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit
def _pair_counts_all_pairs(points, bin_edges):
    '''
    Counts #distances between points in each bin, returns counts list
    '''
    N = points.shape[0]
    n_bins = bin_edges.shape[0] - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(N):
        for j in range(i + 1, N):
            d = _euclidean_distance_2d(points, i, j)
            b = _find_bin(d, bin_edges)
            if b >= 0:
                counts[b] += 1
    return counts


@njit
def _link_counts_from_edges(points, edges, bin_edges):
    '''
    Counts #links for each bin, returns counts list
    '''
    n_bins = bin_edges.shape[0] - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    for e in range(edges.shape[0]):
        i, j = edges[e, 0], edges[e, 1]
        d = _euclidean_distance_2d(points, i, j)
        b = _find_bin(d, bin_edges)
        if b >= 0:
            counts[b] += 1
    return counts


def _bins_from_distances(d, n_bins, method, r_log_floor=0.2):
    if method == "quantile":
        bin_edges = np.unique(np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(bin_edges) < 5:
            raise ValueError("Too few unique distance-bin edges; use 'linear'.")
    elif method == "linear":
        bin_edges = np.linspace(0.0, d.max(), n_bins + 1)
    elif method == "log":
        bin_edges = np.concatenate(([0.0], np.geomspace(r_log_floor, d.max(), n_bins)))
    else:
        raise ValueError("method must be 'quantile', 'linear' or 'log'")
    bin_edges[0] = 0.0
    bin_edges[-1] = bin_edges[-1] * (1.0 + 1e-12) + 1e-12
    return bin_edges


def make_distance_bins(points, n_bins=80, method="quantile",
                       n_sample_pairs=1_000_000, seed=123, r_log_floor=0.2):
    """Radial bin edges from sampled pair distances (quantile, linear or log)."""
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    i = rng.integers(0, N, size=n_sample_pairs)
    j = rng.integers(0, N, size=n_sample_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    d = np.sqrt((points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2)
    return _bins_from_distances(d, n_bins, method, r_log_floor=r_log_floor)


def make_group_distance_bins(points, group, n_bins=80, method="quantile",
                             n_sample_pairs=1_000_000, seed=123, r_log_floor=0.2):
    """Radial bin edges from pair distances within a node subset (group mask)."""
    idx = np.where(group)[0]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, idx.size, size=n_sample_pairs)
    j = rng.integers(0, idx.size, size=n_sample_pairs)
    mask = i != j
    pi, pj = idx[i[mask]], idx[j[mask]]
    d = np.sqrt((points[pi, 0] - points[pj, 0]) ** 2
                + (points[pi, 1] - points[pj, 1]) ** 2)
    return _bins_from_distances(d, n_bins, method, r_log_floor=r_log_floor)


def summarize_bin_stats(bin_edges, pair_counts, link_counts):
    r_left, r_right = bin_edges[:-1], bin_edges[1:]
    r_mid = 0.5 * (r_left + r_right)
    with np.errstate(divide="ignore", invalid="ignore"):
        empirical_p = link_counts / pair_counts
    empirical_p = np.where(pair_counts > 0, empirical_p, np.nan)
    return {
        "r_mid": r_mid,
        "pair_counts": pair_counts,
        "link_counts": link_counts,
        "empirical_p": empirical_p,
    }


def empirical_bin_stats(points, edges, n_bins=80, bin_method="quantile",
                        n_sample_pairs=1_000_000, seed=123, r_log_floor=0.2):
    bin_edges = make_distance_bins(points, n_bins=n_bins, method=bin_method,
                                   n_sample_pairs=n_sample_pairs, seed=seed,
                                   r_log_floor=r_log_floor)
    pair_counts = _pair_counts_all_pairs(points, bin_edges)
    link_counts = _link_counts_from_edges(points, edges, bin_edges)
    return bin_edges, summarize_bin_stats(bin_edges, pair_counts, link_counts)


# ------------------------------------------------------- density / non-homogeneous

@njit
def _pair_counts_grouped(points, group, bin_edges):
    N = points.shape[0]
    n_bins = bin_edges.shape[0] - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(N):
        if not group[i]:
            continue
        for j in range(i + 1, N):
            if not group[j]:
                continue
            b = _find_bin(_euclidean_distance_2d(points, i, j), bin_edges)
            if b >= 0:
                counts[b] += 1
    return counts


@njit
def _link_counts_grouped(points, edges, group, bin_edges):
    n_bins = bin_edges.shape[0] - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    for e in range(edges.shape[0]):
        i, j = edges[e, 0], edges[e, 1]
        if not group[i] or not group[j]:
            continue
        b = _find_bin(_euclidean_distance_2d(points, i, j), bin_edges)
        if b >= 0:
            counts[b] += 1
    return counts


def local_density(points, radius):
    """Local node density: number of network vertices within `radius` km of each
    vertex, evaluated with a KD-tree range query (cKDTree.query_ball_point)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    counts = np.array([len(tree.query_ball_point(p, radius)) for p in points],
                      dtype=np.float64)
    return counts


def density_split_bin_stats(points, edges, group, bin_edges):
    """Radial bin stats restricted to pairs (i, j) with group[i] and group[j]."""
    pair_counts = _pair_counts_grouped(points, group, bin_edges)
    link_counts = _link_counts_grouped(points, edges, group, bin_edges)
    return summarize_bin_stats(bin_edges, pair_counts, link_counts)


# ---------------------------------------------------------------- kernels

# Each model: (n_shape_params, shape_fn(pars, r), pars_from_z(z), initial(z_exp))
# The global amplitude is always handled as lambda (sparse regime): p = (lam/N) * shape.
MODELS = {}


def _register(name, n_params, shape_fn, pars_from_z, initial):
    MODELS[name] = (n_params, shape_fn, pars_from_z, initial)


def _shape_exponential(pars, r):
    return np.exp(-r / pars["ell"])


def _shape_gaussian(pars, r):
    return np.exp(-(r / pars["ell"]) ** 2)


def _shape_powerlaw_cutoff(pars, r):
    return (1.0 + r / pars["r0"]) ** (-pars["alpha"])


def _shape_stretched(pars, r):
    return np.exp(-(r / pars["ell"]) ** pars["beta"])


def _shape_exp_power(pars, r):
    return np.exp(-r / pars["ell"]) * (1.0 + r / pars["r0"]) ** (-pars["alpha"])


def _pars_exp_from_z(z):
    return {"lam": float(np.exp(z[0])), "ell": float(np.exp(z[1]))}


def _pars_gauss_from_z(z):
    return {"lam": float(np.exp(z[0])), "ell": float(np.exp(z[1]))}


def _pars_pl_from_z(z):
    return {"lam": float(np.exp(z[0])), "r0": float(np.exp(z[1])),
            "alpha": float(np.exp(z[2]))}


def _pars_stretched_from_z(z):
    return {"lam": float(np.exp(z[0])), "ell": float(np.exp(z[1])),
            "beta": float(np.exp(z[2]))}


def _pars_exp_power_from_z(z):
    return {"lam": float(np.exp(z[0])), "ell": float(np.exp(z[1])),
            "r0": float(np.exp(z[2])), "alpha": float(np.exp(z[3]))}


_register("exponential", 1, _shape_exponential, _pars_exp_from_z, None)
_register("gaussian", 1, _shape_gaussian, _pars_gauss_from_z, None)
_register("powerlaw_cutoff", 2, _shape_powerlaw_cutoff, _pars_pl_from_z, None)
_register("stretched_exponential", 2, _shape_stretched, _pars_stretched_from_z, None)
_register("exp_power", 3, _shape_exp_power, _pars_exp_power_from_z, None)


def model_prob(r, z, N, model):
    """Connection probability p(r) at bin centers for unconstrained params z."""
    _, shape_fn, pars_from_z, _ = MODELS[model]
    pars = pars_from_z(z)
    shape = shape_fn(pars, r)
    return (pars["lam"] / N) * shape, pars


def _binomial_nll(z, r, pair_counts, link_counts, N, min_pairs, eps, model):
    p, _ = model_prob(r, z, N, model)
    mask = pair_counts >= min_pairs
    p = p[mask]
    if np.any(~np.isfinite(p)):
        return 1e100
    # Clip BEFORE taking logs so that m=0 bins with underflowed p (0.0) give
    # m*log(eps)=0 instead of 0*log(0)=nan.
    p = np.clip(p, eps, 1.0 - eps)
    n = pair_counts[mask].astype(float)
    m = link_counts[mask].astype(float)
    ll = np.sum(m * np.log(p) + (n - m) * np.log(1.0 - p))
    return -ll if np.isfinite(ll) else 1e100


def _bounds(model, N):
    """Finite bounds on the unconstrained params to avoid degenerate scale/amplitude."""
    lam_lo, lam_hi = np.log(0.01), np.log(max(2.0 * N, 100.0))
    ell = (np.log(0.5), np.log(5000.0))
    r0 = (np.log(0.5), np.log(5000.0))
    alpha = (np.log(0.05), np.log(15.0))
    beta = (np.log(0.05), np.log(3.0))
    if model in ("exponential", "gaussian"):
        return [lam_lo, lam_hi], [ell[0], ell[1]]
    if model == "powerlaw_cutoff":
        return [lam_lo, lam_hi], [r0[0], r0[1]], [alpha[0], alpha[1]]
    if model == "stretched_exponential":
        return [lam_lo, lam_hi], [ell[0], ell[1]], [beta[0], beta[1]]
    if model == "exp_power":
        return [lam_lo, lam_hi], [ell[0], ell[1]], [r0[0], r0[1]], [alpha[0], alpha[1]]
    raise ValueError(f"Unknown model: {model}")


def _clamp_to_bounds(z, bounds):
    z = np.asarray(z, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        z[i] = min(max(z[i], lo), hi)
    return z


def _initial_z(r, pair_counts, link_counts, N, model, eps=1e-12):
    """Log-linear initial guess for the exponential part, weak tail otherwise."""
    mask = (pair_counts > 0) & (link_counts > 0)
    if np.sum(mask) < 3:
        return np.array([np.log(5.0), np.log(100.0)])

    p_hat = (link_counts[mask] + 0.5) / (pair_counts[mask] + 1.0)
    y = np.log(np.maximum(p_hat, eps))
    x = r[mask]
    w = pair_counts[mask].astype(float)
    X = np.column_stack([np.ones_like(x), x])
    XtW = X.T * w
    beta = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]

    ell0 = max(-1.0 / beta[1], 1e-3) if beta[1] < 0 else 100.0
    lam0 = max(N * np.exp(beta[0]), 1e-4)

    r0 = max(np.median(r), 1e-6)
    if model == "exponential" or model == "gaussian":
        return np.array([np.log(lam0), np.log(ell0)])
    if model == "powerlaw_cutoff":
        return np.array([np.log(lam0), np.log(r0), np.log(0.5)])
    if model == "stretched_exponential":
        return np.array([np.log(lam0), np.log(ell0), np.log(0.5)])
    if model == "exp_power":
        return np.array([np.log(lam0), np.log(ell0), np.log(r0), np.log(0.1)])
    raise ValueError(f"Unknown model: {model}")


def fit_kernel(bin_stats, N, model="exponential", min_pairs=1000, eps=1e-12,
               method="L-BFGS-B"):
    r = bin_stats["r_mid"]
    pair_counts = bin_stats["pair_counts"]
    link_counts = bin_stats["link_counts"]

    bounds = _bounds(model, N)
    z0 = _clamp_to_bounds(_initial_z(r, pair_counts, link_counts, N, model, eps), bounds)

    result = minimize(
        _binomial_nll, z0,
        args=(r, pair_counts, link_counts, N, min_pairs, eps, model),
        method=method,
        bounds=bounds if method == "L-BFGS-B" else None,
        options={"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-8},
    )

    p_fit, pars = model_prob(r, result.x, N, model)
    nll = _binomial_nll(result.x, r, pair_counts, link_counts, N, min_pairs, eps, model)
    k = len(result.x)
    aic = 2 * k + 2 * nll
    bic = k * np.log(np.sum(pair_counts)) + 2 * nll

    return {
        "model": model,
        "success": bool(result.success),
        "nll": float(nll),
        "aic": float(aic),
        "bic": float(bic),
        "n_params": int(k),
        "params_hat": pars,
        "p_fit_bins": p_fit,
    }
