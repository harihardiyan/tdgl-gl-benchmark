# TDGL Benchmark Suite – API Documentation

This document describes the public API of the TDGL Ginzburg–Landau Benchmark Suite.  
All functions are implemented in pure JAX and designed for scientific reproducibility.

---

# 1. tdgl_core.core

## 1.1 State
Dataclass representing the TDGL dynamical fields.

Fields:
- psi : (ny, nx, 2) real-imag representation of ψ
- A   : (ny, nx, 2) vector potential
- mu  : (ny, nx) scalar potential

## 1.2 Params
Dataclass containing physical and numerical parameters.

Fields:
- nx, ny : grid size
- dx, dy : grid spacing
- kappa : GL parameter
- gamma_psi, gamma_A : relaxation constants
- sigma_n : normal conductivity
- J_ext : external current
- a : GL coefficient field
- H_ext : external magnetic field

---

## 1.3 to_complex(phi)
Convert real-imag array → complex array.

## 1.4 to_real(z)
Convert complex array → real-imag array.

---

## 1.5 grad_x_neumann(f, dx)
Centered finite difference with Neumann BC.

## 1.6 grad_y_neumann(f, dy)
Centered finite difference with Neumann BC.

## 1.7 curl_A_neumann(A, dx, dy)
Compute Bz = ∂x Ay − ∂y Ax.

---

## 1.8 covariant_grad_psi_link(psi, A, dx, dy)
Gauge-invariant covariant gradient using link variables.

---

## 1.9 energy_density(state, params)
Compute GL energy density.

## 1.10 total_energy(state, params)
Integrate energy density over domain.

---

## 1.11 solve_mu_poisson(mu0, params, n_iter, omega)
SOR Poisson solver for scalar potential μ.

---

## 1.12 tdgl_rhs(state, params)
Compute TDGL right-hand side.

## 1.13 tdgl_step(state, params, dt, n_mu_iter)
One TDGL time step.

---

## 1.14 init_params_basic(...)
Initialize domain, geometry, and physical parameters.

## 1.15 init_state_random(params, noise, key)
Random initial ψ with noise.

---

# 2. tdgl_core.observables

## 2.1 detect_vortices_phase_winding(psi)
Detect vortices via phase winding.

## 2.2 radial_profile(psi_abs2, center, dr)
Compute radial |ψ|² profile.

## 2.3 gl_profile(r, xi)
Analytic GL vortex profile.

## 2.4 fit_coherence_length(r, psi_r)
Fit GL profile to extract ξ.

---

## 2.5 relax_for_field(B0, ...)
Relax TDGL system at a given external field.

## 2.6 run_H_scan(H_vals, ...)
Run TDGL relaxation over multiple fields.

---

## 2.7 poly_fit(x, y, deg)
Least-squares polynomial fit.

## 2.8 eval_poly(coeffs, x)
Evaluate polynomial.

---

## 2.9 vortex_density(count, params)
Compute vortex density per unit area.

---

# 3. tdgl_core.utils

## 3.1 make_grid(nx, ny, Lx, Ly)
Generate coordinate grids.

## 3.2 set_seed(seed)
Create PRNG key.

---

# End of API
