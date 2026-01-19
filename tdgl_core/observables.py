# tdgl_core/observables.py

import jax
import jax.numpy as jnp
from .core import (
    State, Params, to_complex, total_energy,
    tdgl_step, init_params_basic, init_state_random
)

# ============================================================
# 1. Vortex detection (phase winding)
# ============================================================

def detect_vortices_phase_winding(psi_c: jnp.ndarray) -> jnp.ndarray:
    phase = jnp.angle(psi_c)
    ny, nx = phase.shape

    dpx = jnp.diff(phase, axis=1)
    dpy = jnp.diff(phase, axis=0)

    dpx = (dpx + jnp.pi) % (2 * jnp.pi) - jnp.pi
    dpy = (dpy + jnp.pi) % (2 * jnp.pi) - jnp.pi

    dpx_ij   = dpx[:-1, :]
    dpx_ip1j = dpx[1:, :]
    dpy_ij   = dpy[:, :-1]
    dpy_ijp1 = dpy[:, 1:]

    winding = dpx_ij + dpy_ijp1 - dpx_ip1j - dpy_ij
    n = jnp.round(winding / (2 * jnp.pi)).astype(jnp.int32)

    mask = jnp.zeros((ny, nx), dtype=jnp.int32)
    mask = mask.at[:-1, :-1].set(n)
    return mask


# ============================================================
# 2. Radial profile (JAX-safe, NO BOOLEAN INDEXING)
# ============================================================

def radial_profile(psi_abs2: jnp.ndarray,
                   center: tuple,
                   dr: float = 0.1,
                   r_max: float = None):

    ny, nx = psi_abs2.shape
    cy, cx = center

    # Coordinate grid
    y = jnp.arange(ny)
    x = jnp.arange(nx)
    Y, X = jnp.meshgrid(y, x, indexing="ij")

    # Radial distance
    r = jnp.sqrt((X - cx)**2 + (Y - cy)**2)

    # Max radius
    if r_max is None:
        r_max = r.max()

    # Bin edges
    bins = jnp.arange(0, r_max + dr, dr)

    # Integer bin index for each pixel
    idx = jnp.digitize(r, bins) - 1

    # Flatten for segment ops
    idx_flat = idx.reshape(-1)
    vals_flat = psi_abs2.reshape(-1)

    nbins = len(bins)

    # Sum per bin
    sums = jax.ops.segment_sum(vals_flat, idx_flat, nbins)

    # Count per bin
    counts = jax.ops.segment_sum(jnp.ones_like(vals_flat), idx_flat, nbins)

    # Avoid division by zero
    profile = jnp.where(counts > 0, sums / counts, 0.0)

    return bins, profile


# ============================================================
# 3. GL analytic profile + coherence length fit
# ============================================================

def gl_profile(r, xi):
    return jnp.tanh(r / (jnp.sqrt(2) * xi))**2


def fit_coherence_length(r, psi_r):
    def loss_fn(xi):
        return jnp.mean((gl_profile(r, xi) - psi_r)**2)

    xi0 = 1.0
    xi_opt = jax.scipy.optimize.minimize(loss_fn, xi0).x
    return float(xi_opt)


# ============================================================
# 4. H-scan utilities
# ============================================================

def relax_for_field(B0,
                    nx=64, ny=64,
                    Lx=20.0, Ly=20.0,
                    kappa=2.0,
                    n_relax=1500,
                    dt=0.01,
                    n_mu_iter=80,
                    noise=0.3,
                    seed=0):

    key = jax.random.PRNGKey(seed)
    params = init_params_basic(nx=nx, ny=ny, Lx=Lx, Ly=Ly,
                               kappa=kappa, B0=B0, J_ext=0.0)
    state = init_state_random(params, noise=noise, key=key)

    def body(s, _):
        return tdgl_step(s, params, dt, n_mu_iter), None

    state, _ = jax.lax.scan(body, state, jnp.arange(n_relax))

    psi_c = to_complex(state.psi)
    dens = jnp.abs(psi_c)**2
    E = total_energy(state, params)
    vort_mask = detect_vortices_phase_winding(psi_c)

    vort_pos = jnp.sum(vort_mask == 1)
    vort_neg = jnp.sum(vort_mask == -1)
    var_dens = jnp.var(dens)

    return {
        "state": state,
        "params": params,
        "dens": dens,
        "energy": E,
        "vort_mask": vort_mask,
        "vort_pos": vort_pos,
        "vort_neg": vort_neg,
        "indicator": var_dens,
    }


def run_H_scan(H_vals,
               nx=64, ny=64,
               Lx=20.0, Ly=20.0,
               kappa=2.0,
               n_relax=1500,
               dt=0.01,
               n_mu_iter=80,
               noise=0.3):

    energies = []
    indicators = []
    vort_pos = []
    vort_neg = []
    samples = []

    for i, B0 in enumerate(H_vals):
        out = relax_for_field(
            B0,
            nx=nx, ny=ny,
            Lx=Lx, Ly=Ly,
            kappa=kappa,
            n_relax=n_relax,
            dt=dt,
            n_mu_iter=n_mu_iter,
            noise=noise,
            seed=1000 + i
        )

        energies.append(out["energy"])
        indicators.append(out["indicator"])
        vort_pos.append(out["vort_pos"])
        vort_neg.append(out["vort_neg"])

        if i in (0, len(H_vals)//2, len(H_vals)-1):
            samples.append(out)

    return {
        "H_vals": H_vals,
        "energies": jnp.array(energies),
        "indicators": jnp.array(indicators),
        "vort_pos": jnp.array(vort_pos),
        "vort_neg": jnp.array(vort_neg),
        "samples": samples,
    }


# ============================================================
# 5. Polynomial fitting
# ============================================================

def poly_fit(x, y, deg=1):
    X = jnp.stack([x**k for k in range(deg + 1)], axis=-1)
    XT = X.T
    A = XT @ X
    b = XT @ y
    return jnp.linalg.solve(A, b)


def eval_poly(coeffs, x):
    return sum(coeffs[k] * x**k for k in range(len(coeffs)))


# ============================================================
# 6. Vortex density
# ============================================================

def vortex_density(vort_count, params: Params):
    area = params.dx * params.nx * params.dy * params.ny
    return vort_count / area
