"""Landau-Ginzburg field-theory covariance for embedded networks.

Bridges empirical graph kernels to statistical-field-theory covariance via the
explicit quadratic operator (precision matrix) of a Gaussian scalar field:

    F[phi] = int dx [ kappa/2 (grad phi)^2 + r0/2 phi^2 + u/4 phi^4 ]

with u = 0 (Gaussian baseline). On a regular 2D grid with spacing h the
precision operator is the symmetric matrix

    Q = (kappa / h^2) L + r0 I

where L is the 5-point graph Laplacian; the covariance is G = T Q^{-1}. We build
Q explicitly and solve G = T Q^{-1} numerically, radially interpolate it onto the
empirical bin centres, and fit the operator coefficients (amplitude A, mass r0)
to the empirical kernel p(r). The correlation length is xi = sqrt(kappa / r0).

Non-homogeneous level: the mass is promoted to a spatially varying coefficient
over coarse regions, r(x) = r0 (1 + eps m(x)) with m(x) the density anomaly.
Because a spatially varying mass does not change the *radially averaged*
covariance to first order, the non-homogeneous level is implemented over two
coarse regions (dense / sparse): the homogeneous operator is fitted to each
region's kernel separately (giving r_dense, r_sparse), and the two masses are
inverted for (r0, eps).
"""
import numpy as np
from functools import lru_cache
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.optimize import minimize


# ------------------------------------------------------------------ operator

@lru_cache(maxsize=None)
def grid_laplacian(n):
    """Graph Laplacian of an n x n grid (4-nearest neighbours), sparse CSC."""
    N = n * n
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    for ix in range(n):
        for iy in range(n):
            i = ix * n + iy
            deg = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < n and 0 <= jy < n:
                    add(i, jx * n + jy, -1.0)
                    deg += 1
            add(i, i, float(deg))

    return sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def make_grid_geometry(points, n=None, h_target=None):
    """Square grid covering the bounding box of `points`; spacing h (km).

    If `h_target` is given the grid side `n` is derived so the actual spacing
    satisfies h <= h_target (n = ceil(L / h_target) + 1). Otherwise a fixed
    side `n` is used. A coarse spacing (h larger than the dense-region
    correlation length) cannot resolve sub-h structure, so h_target must be
    chosen below the smallest physical scale of interest.
    """
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    L = max(xmax - xmin, ymax - ymin)
    if n is None:
        if h_target is None:
            raise ValueError("provide either `n` or `h_target`")
        n = int(np.ceil(L / h_target)) + 1
    h = L / (n - 1)
    xs = xmin + h * np.arange(n)
    ys = ymin + h * np.arange(n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()]), h


def build_precision(n, h, kappa, r0):
    """Q = (kappa / h^2) L + r0 I, sparse CSC."""
    L = grid_laplacian(n)
    mass = max(float(r0), 1e-12)
    return ((kappa / h ** 2) * L + mass * sparse.identity(n * n)).tocsc()


# ----------------------------------------------------- radial covariance of Q

def _radialize(d, g, bin_edges, logspace=True):
    """Collapse (distance, value) samples into a radial profile on bin centres.

    Mean `g` over equal radii, then interpolate onto the bin centres. `logspace`
    (log-log interpolation) is used for strictly positive profiles (the Green's
    function) to resolve the steep small-r decay; `logspace=False` (linear
    interpolation) is used for signed corrections such as the perturbative
    delta-G, whose logarithm is undefined. Bins outside the sampled radial
    range are returned as 0 (excluded downstream by the r >= r_min mask).
    """
    n_bins = len(bin_edges) - 1
    r_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    order = np.argsort(d)
    d = d[order]; g = g[order]
    first = np.unique(d, return_index=True)[1]
    d_uniq = d[first]
    g_uniq = np.add.reduceat(g, first) / np.diff(np.append(first, len(d)))

    g_rad = np.zeros(n_bins)
    inside = (r_mid >= d_uniq[0]) & (r_mid <= d_uniq[-1])
    if inside.any():
        if logspace:
            with np.errstate(divide="ignore", invalid="ignore"):
                g_rad[inside] = np.exp(np.interp(np.log(r_mid[inside]),
                                                 np.log(d_uniq), np.log(g_uniq)))
        else:
            g_rad[inside] = np.interp(r_mid[inside], d_uniq, g_uniq)
    return g_rad


def _radialize_weighted(d, g, bin_edges):
    """Collapse (distance, value) samples into a multiplicity-weighted profile.

    Averages `g` over all samples whose distance falls in each bin, weighted by
    the sample multiplicity (each sample is one source->target pair), so the
    result is the pair-weighted radial average -- the model analogue of the
    empirical p(r) = links / pairs. Bins with no sample (e.g. r < h, where no
    lattice shell exists) are filled by log-log interpolation from the nearest
    populated bins, keeping the profile defined at every bin centre.
    """
    n_bins = len(bin_edges) - 1
    r_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        sums, _ = np.histogram(d, bins=bin_edges, weights=g)
        counts, _ = np.histogram(d, bins=bin_edges)
    g_rad = np.full(n_bins, np.nan)
    ok = counts > 0
    g_rad[ok] = sums[ok] / counts[ok]

    filled = np.isfinite(g_rad)
    if filled.sum() < 2:
        return np.nan_to_num(g_rad, nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_rad[~filled] = np.exp(np.interp(np.log(r_mid[~filled]),
                                          np.log(r_mid[filled]),
                                          np.log(g_rad[filled])))
    return g_rad


def _interior_mask(n, margin):
    """Boolean mask over an n x n grid marking sites away from the boundary.

    Returns a 1D boolean array (length n*n), True for sites whose row and
    column indices both lie in [margin, n - margin). Used to draw source sites
    far from the open grid boundary, where the finite-lattice Green's function
    is suppressed relative to the infinite plane.
    """
    ii = np.arange(n)
    inside = (ii >= margin) & (ii < n - margin)
    return np.outer(inside, inside).ravel()


def radial_covariance(Q, coords, bin_edges, temperature=1.0, src=None,
                      weighted=False):
    """Radially average G = T Q^{-1} at the empirical bin centres.

    Solves Q g_s = T e_s for each source column (LU factorization reused) and
    collects the discrete Green's-function samples over all source->target
    pairs. With `weighted=False` the samples are collapsed by log-log
    interpolation onto the bin centres (smoothing the quantized lattice radii
    h*sqrt(i^2+j^2)); with `weighted=True` they are binned directly with
    multiplicity weighting (the pair-weighted average), matching how the
    empirical kernel is estimated. Self-pairs (d = 0) are excluded, matching
    the empirical kernel which never counts them.
    """
    N = coords.shape[0]
    if src is None:
        src = np.arange(N)
    src = np.asarray(src, dtype=np.int64)
    lu = spla.splu(Q.tocsc())

    ds, gs = [], []
    if src.size == N:
        # every site is a source: avoid a dense N x N RHS, solve column by column
        for s in src:
            rhs = np.zeros(N); rhs[s] = temperature
            g = lu.solve(rhs)                   # column s of G = T Q^{-1}
            d = np.sqrt(((coords - coords[s]) ** 2).sum(axis=1))
            ok = d > 0.0
            ds.append(d[ok]); gs.append(g[ok])
    else:
        rhs = np.zeros((N, src.size))
        rhs[src, np.arange(src.size)] = temperature
        G = lu.solve(rhs)                       # batched solve over all source columns
        for i, s in enumerate(src):
            d = np.sqrt(((coords - coords[s]) ** 2).sum(axis=1))
            ok = d > 0.0
            ds.append(d[ok]); gs.append(G[ok, i])
    d = np.concatenate(ds); g = np.concatenate(gs)
    if weighted:
        return _radialize_weighted(d, g, bin_edges)
    return _radialize(d, g, bin_edges)


# ------------------------------------------------ non-homogeneous / perturbative

def grid_density_anomaly(points, coords, radius):
    """Standardized density anomaly m(x) evaluated on the lattice sites.

    Uses the *same* estimator as kernel.local_density: the number of network
    vertices within `radius` km of a query position, evaluated with a KD-tree
    range query (cKDTree.query_ball_point). Here the queries are the grid
    sites (`coords`) rather than the nodes, so the node-level density (used
    for the empirical dense/sparse split) and the grid mass field are the
    same quantity. Returns m(x) = (rho - mean)/std over grid sites.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    rho = np.array([len(tree.query_ball_point(p, radius)) for p in coords],
                   dtype=np.float64)
    return density_anomaly(rho)


def build_precision_inhomogeneous(n, h, kappa, r0, eps, m_grid):
    """Non-homogeneous precision Q = (kappa/h^2) L + diag(r0 (1 + eps m(x))).

    Generalizes build_precision to a spatially varying mass r(x) = r0(1 + eps m).
    """
    L = grid_laplacian(n)
    mass = np.maximum(r0 * (1.0 + eps * np.asarray(m_grid, dtype=float)), 1e-12)
    return ((kappa / h ** 2) * L + sparse.diags(mass)).tocsc()


def radial_covariance_perturbative(Q0_lu, coords, bin_edges, temperature,
                                   m_grid, r0, eps, src):
    """First-order (in eps) expansion of the non-homogeneous radial covariance.

    Split Q = Q0 + eps M with M = diag(r0 m_grid) and Q0 = Q(eps=0); then

        G = T Q^{-1} = G0 + delta G + O(eps^2),
        delta G = -eps T Q0^{-1} M Q0^{-1}.

    For a source column s, with g0 = T Q0^{-1} e_s, the temperature cancels and

        delta G e_s = -eps Q0^{-1} (r0 m_grid * g0).

    Q0_lu is the sparse LU factorization of Q0 (reused: only one extra solve per
    source is needed for any eps). Returns (g0_rad, dg_rad, g_rad) with
    g_rad = g0_rad + dg_rad the O(eps) approximation to the full profile.
    """
    N = coords.shape[0]
    src = np.asarray(src, dtype=np.int64)

    ds, g0s, dgs = [], [], []
    for s in src:
        rhs = np.zeros(N); rhs[s] = temperature
        g0 = Q0_lu.solve(rhs)                     # column s of G0
        dg = -eps * Q0_lu.solve(r0 * m_grid * g0) # column s of delta G
        d = np.sqrt(((coords - coords[s]) ** 2).sum(axis=1))
        ok = d > 0.0
        ds.append(d[ok]); g0s.append(g0[ok]); dgs.append(dg[ok])
    d = np.concatenate(ds)
    g0 = np.concatenate(g0s); dg = np.concatenate(dgs)

    g0_rad = _radialize(d, g0, bin_edges, logspace=True)
    dg_rad = _radialize(d, dg, bin_edges, logspace=False)
    return g0_rad, dg_rad, g0_rad + dg_rad


def _radial_covariance_class(Q0_lu, coords, bin_edges, temperature, m_grid,
                             r0, eps, src, tgt_mask):
    """Radial profile of G = G0 + delta G restricted to a target site class.

    Sources are drawn from `src`; only target sites where `tgt_mask` holds
    contribute to the radial average. This is the model analogue of the
    empirical density-conditioned kernel (both endpoints in the same region).
    """
    N = coords.shape[0]
    src = np.asarray(src, dtype=np.int64)
    tgt_mask = np.asarray(tgt_mask, dtype=bool)

    ds, gs = [], []
    for s in src:
        rhs = np.zeros(N); rhs[s] = temperature
        g0 = Q0_lu.solve(rhs)
        dg = -eps * Q0_lu.solve(r0 * m_grid * g0)
        g = np.maximum(g0 + dg, 1e-300)   # covariance is non-negative; floor the O(eps) tail
        d = np.sqrt(((coords - coords[s]) ** 2).sum(axis=1))
        ok = (d > 0.0) & tgt_mask
        ds.append(d[ok]); gs.append(g[ok])
    if not ds:
        return np.zeros(len(bin_edges) - 1)
    d = np.concatenate(ds); g = np.concatenate(gs)
    return _radialize(d, g, bin_edges, logspace=True)


def one_loop_mass_shift(Q0_lu, n, r0, u, temperature, n_diag=256, seed=123):
    """One-loop (Hartree/tadpole) mass renormalization from the quartic u phi^4.

    To first order in u the quartic term only shifts the effective mass:

        r_eff = r0 + 3 u <phi^2>_0,   <phi^2>_0 = T * mean(diag(Q0^{-1})).

    The factor 3 follows from the u/4 phi^4 vertex convention (three tadpole
    contractions). Position-independent for the homogeneous baseline, so it
    uniformly renormalizes the correlation length xi = sqrt(kappa / r_eff).
    """
    N = n * n
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n_diag, N), replace=False)
    diag = np.empty(idx.size)
    for k, i in enumerate(idx):
        rhs = np.zeros(N); rhs[i] = 1.0
        diag[k] = Q0_lu.solve(rhs)[i]
    phi2_0 = temperature * diag.mean()
    return float(r0 + 3.0 * u * phi2_0), float(phi2_0)


# ---------------------------------------------------------------------- fit

def _quadratic_loss(z, n, coords, h, bin_edges, src, p, w, mask, kappa,
                    temperature, weighted):
    """Weighted log-log least squares of A * g(r; r0) against p(r)."""
    A = float(np.exp(z[0])); r0 = float(np.exp(z[1]))
    Q = build_precision(n, h, kappa, r0)
    g = radial_covariance(Q, coords, bin_edges, temperature=temperature, src=src,
                          weighted=weighted)
    model = A * g[mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.log(p[mask]) - np.log(model)
    ok = np.isfinite(res)
    if not ok.any():
        return 1e100
    return float(np.mean(w[mask][ok] * res[ok] ** 2) / np.mean(w[mask][ok]))


def fit_quadratic_covariance(coords, h, bin_edges, r_mid, empirical_p, pair_counts,
                             kappa=1.0, temperature=1.0, min_pairs=1000,
                             r_min=0.0, n_src=None, seed=123, amp0=1e-6, r00=1e-3,
                             src=None, weighted=False, source_margin=None):
    """Fit the homogeneous operator coefficients (A, r0) to the empirical kernel.

    The covariance is evaluated from the single central column (deterministic)
    unless `src` is given, or from `n_src` source sites sampled from the grid
    interior when `n_src` is given (ergodic radial average; `source_margin`
    lattice spacings are excluded from each grid edge). `weighted=True` uses
    multiplicity-weighted binning instead of log-log interpolation. Returns the
    discrete model on the masked bins.
    """
    n = int(round(np.sqrt(coords.shape[0])))
    mask = ((pair_counts >= min_pairs) & (empirical_p > 0)
            & np.isfinite(empirical_p) & (r_mid >= r_min))
    if not mask.any():
        raise ValueError("No bins pass the mask (min_pairs / r_min); relax them.")

    p = np.asarray(empirical_p, dtype=float)
    w = pair_counts.astype(float) / pair_counts.mean()

    if src is None:
        if n_src is not None:
            rng = np.random.default_rng(seed)
            if source_margin is not None and source_margin > 0:
                pool = np.where(_interior_mask(n, int(source_margin)))[0]
                if pool.size == 0:
                    pool = np.arange(coords.shape[0])
            else:
                pool = np.arange(coords.shape[0])
            src = rng.choice(pool, size=min(n_src, pool.size), replace=False)
        else:
            src = np.array([coords.shape[0] // 2])   # central column, deterministic

    z0 = [np.log(amp0), np.log(r00)]
    bounds = [(np.log(1e-15), np.log(1.0)), (np.log(1e-8), np.log(10.0))]

    res = minimize(_quadratic_loss, z0,
                   args=(n, coords, h, bin_edges, src, p, w, mask, kappa,
                         temperature, weighted),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-8})

    A = float(np.exp(res.x[0])); r0 = float(np.exp(res.x[1]))

    Q = build_precision(n, h, kappa, r0)
    g = radial_covariance(Q, coords, bin_edges, temperature=temperature, src=src,
                          weighted=weighted)

    r_m = r_mid[mask]
    p_m = p[mask]
    model_m = A * g[mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(p_m); logm = np.log(model_m)
    ok = np.isfinite(logp) & np.isfinite(logm)
    mse_log = float(np.mean((logp[ok] - logm[ok]) ** 2)) if ok.any() else float("nan")
    mse_lin = float(np.mean((p_m - model_m) ** 2))
    ss_res = np.sum((logp[ok] - logm[ok]) ** 2)
    ss_tot = np.sum((logp[ok] - logp[ok].mean()) ** 2)

    return {
        "success": bool(res.success),
        "amplitude": A,
        "r0": r0,
        "xi": float(np.sqrt(kappa / r0)),
        "mse_log": mse_log,
        "mse_lin": mse_lin,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "model_r": r_m,
        "model_bins": model_m,
        "loss": float(res.fun),
        "n_mask": int(mask.sum()),
    }


def _eps_loss(z, Q0_lu, coords, m_grid, r0, temperature, groups):
    """Weighted log-log least squares of A (G0 + delta G) against the class
    kernels; A is shared across classes, eps enters only through the cheap
    delta-G solve (Q0_lu is factorized once)."""
    A = float(np.exp(z[0])); eps = float(z[1])
    tot = 0.0; wsum = 0.0
    for g in groups:
        G = _radial_covariance_class(Q0_lu, coords, g["bin_edges"], temperature,
                                     m_grid, r0, eps, g["src"], g["tgt_mask"])
        model = A * G
        p = g["empirical_p"]; mask = g["mask"]; w = g["w"]
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.log(p[mask]) - np.log(model[mask])
        ok = np.isfinite(res) & (model[mask] > 1e-200)
        if not ok.any():
            return 1e100
        tot += float(np.sum(w[mask][ok] * res[ok] ** 2))
        wsum += float(np.sum(w[mask][ok]))
    return tot / wsum if wsum > 0 else 1e100


def fit_perturbative_eps(Q0_lu, coords, m_grid, r0, kappa, temperature, groups,
                         amp0=1e-6, eps0=0.0, eps_bounds=(-3.0, 3.0)):
    """Fit (A, eps) of the O(eps) non-homogeneous covariance to density-class kernels.

    The homogeneous mass r0 (hence Q0) is held fixed; only the amplitude A and
    the inhomogeneity coupling eps are tuned, so a single LU factorization of
    Q0 is reused across all eps evaluations. `groups` is a list of dicts, one
    per density class, each with keys:

        bin_edges, empirical_p, pair_counts (radial binning of that class),
        mask (bins passing min_pairs / r_min / p>0), w (normalized weights),
        tgt_mask (grid sites in the class), src (source-site indices).

    Returns the fitted (A, eps), the per-class model profiles, and fit metrics.
    """
    for g in groups:
        g["w"] = (g["pair_counts"].astype(float) / g["pair_counts"].mean()
                  if g["pair_counts"].mean() > 0 else np.ones_like(g["pair_counts"], dtype=float))

    z0 = [np.log(amp0), eps0]
    bounds = [(np.log(1e-15), np.log(1.0)), (eps_bounds[0], eps_bounds[1])]
    res = minimize(_eps_loss, z0,
                   args=(Q0_lu, coords, m_grid, r0, temperature, groups),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6})

    A = float(np.exp(res.x[0])); eps = float(res.x[1])

    models = []
    for g in groups:
        G = _radial_covariance_class(Q0_lu, coords, g["bin_edges"], temperature,
                                     m_grid, r0, eps, g["src"], g["tgt_mask"])
        model = A * G
        p = g["empirical_p"]; mask = g["mask"]
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.log(p[mask]); logm = np.log(model[mask])
        ok = np.isfinite(logp) & np.isfinite(logm) & (model[mask] > 1e-200)
        mse_log = float(np.mean((logp[ok] - logm[ok]) ** 2)) if ok.any() else float("nan")
        models.append({"model_r": g["r_mid"][mask],
                       "model_bins": model[mask],
                       "mse_log": mse_log})

    return {
        "success": bool(res.success),
        "amplitude": A,
        "eps": eps,
        "loss": float(res.fun),
        "models": models,
    }


# ------------------------------------------------------------------ observables

def density_anomaly(density):
    """Standardized density anomaly m(x) = (rho - mean) / std."""
    rho = np.asarray(density, dtype=float)
    std = rho.std()
    if std <= 0:
        return np.zeros_like(rho)
    return (rho - rho.mean()) / std


def infer_nonhomogeneous_coefficients(r_dense, r_sparse, m_dense, m_sparse, kappa=1.0):
    """Invert r_g = kappa / xi_g^2 = r0 (1 + eps m_g) for (r0, eps).

    Returns the inferred non-homogeneous potential coefficients. If the two
    regions have identical mass the system is degenerate and eps is reported as 0.
    """
    rd = float(r_dense); rs = float(r_sparse)
    if abs(m_dense - m_sparse) < 1e-12:
        return {"r0": 0.5 * (rd + rs), "eps": 0.0}
    eps = (rd - rs) / (m_dense * rs - m_sparse * rd)
    r0 = rd / (1.0 + eps * m_dense)
    if r0 <= 0:
        r0 = rs / (1.0 + eps * m_sparse)
    return {
        "r0": float(r0),
        "eps": float(eps),
        "xi_dense": float(np.sqrt(kappa / max(rd, 1e-12))),
        "xi_sparse": float(np.sqrt(kappa / max(rs, 1e-12))),
    }
