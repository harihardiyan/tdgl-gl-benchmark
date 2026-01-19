

TDGL Ginzburg–Landau Benchmark Suite (JAX)
![License: MIT](LICENSE)
![Python]()
![JAX]()
![Colab]()

A modular, reproducible, and research‑grade implementation of the time‑dependent Ginzburg–Landau (TDGL) equations for 2D superconductors, built using JAX for high‑performance automatic differentiation, JIT compilation, and GPU/TPU acceleration.

This repository provides:

- A clean TDGL core solver (tdgl_core/core.py)
- A full observables suite (tdgl_core/observables.py)
- Colab‑ready notebooks demonstrating single‑vortex, multi‑vortex, H‑scan, and physical observables
- A reproducible benchmark pipeline suitable for research, teaching, and publication

---

✨ Key Features

TDGL Core
- PyTree‑based State and Params for clean JAX integration  
- Gauge‑invariant link‑variable covariant gradient  
- Full GL energy functional  
- μ‑Poisson solver (SOR) for gauge fixing  
- JIT‑accelerated TDGL time stepping  
- Uniform geometry initialization  

Observables
- Vortex detection via phase winding  
- Radial vortex profiles  
- Analytic GL fit for coherence length \( \xi \)  
- H‑field scans (Meissner → vortex → high‑field)  
- Vortex count, density, and variance indicators  
- Polynomial fits for energy vs vortex count  
- Numerical estimates of \( H{c1} \) and \( H{c2} \)

Notebooks
- Single vortex benchmark  
- Multi‑vortex (64×64 and 128×128)  
- H‑scan and observables suite  
- Energy vs vortex count analysis  

---

🧠 Scientific Background

The TDGL equations describe the dynamics of the superconducting order parameter  
\( \psi(\mathbf{r}, t) \) and vector potential \( \mathbf{A}(\mathbf{r}, t) \):

Time‑dependent Ginzburg–Landau equations

\[
\frac{\partial \psi}{\partial t}
= -\gamma_\psi \frac{\delta F}{\delta \psi^\*},
\qquad
\frac{\partial \mathbf{A}}{\partial t}
= -\gammaA \left( \frac{\delta F}{\delta \mathbf{A}} + \mathbf{J}n \right)
\]

where the GL free energy density is:

\[
f = a|\psi|^2 + \frac{1}{2}|\psi|^4
+ |(\nabla - i\mathbf{A})\psi|^2
+ \kappa^2 (B - H_\text{ext})^2.
\]

The normal current is:

\[
\mathbf{J}n = -\sigman \nabla \mu.
\]

This repository implements these equations using gauge‑invariant link variables,  
Neumann boundary conditions, and a Poisson solver for \( \mu \).

---

🚫 What This Project Does Not Claim

To maintain academic integrity, this repository does not claim:

- To be a full microscopic model (e.g., Bogoliubov–de Gennes, Eilenberger, Usadel)
- To simulate quantum fluctuations or microscopic pairing mechanisms  
- To reproduce experimental vortex patterns quantitatively without calibration  
- To implement 3D TDGL or anisotropic materials (yet)  
- To provide production‑grade device optimization pipelines  
- To replace specialized TDGL packages (e.g., GL‑GPU, SVIRL)  

This project is a clean, modular, and reproducible research benchmark suite,  
not a full superconducting device simulator.

---

📦 Installation

`bash
git clone https://github.com/harihardiyan/tdgl-gl-benchmark.git
cd tdgl-gl-benchmark
pip install -r requirements.txt
`

---

🧪 Usage Example

`python
from tdgl_core import (
    initparamsbasic,
    initstaterandom,
    tdgl_step,
    total_energy,
    detectvorticesphase_winding
)

params = initparamsbasic(nx=64, ny=64, B0=0.1)
state = initstaterandom(params)

for _ in range(1000):
    state = tdgl_step(state, params, dt=0.01)

psi = to_complex(state.psi)
vortices = detectvorticesphase_winding(psi)
energy = total_energy(state, params)
`

---

📚 Documentation

See api.md for full API documentation of:

- tdgl_core.core
- tdgl_core.observables
- tdgl_core.utils

---

📊 Citation

If you use this repository in academic work, please cite:

`
Hari, et al. (2025).
TDGL Ginzburg–Landau Benchmark Suite (JAX).
https://github.com/<yourname>/tdgl-gl-benchmark
`

---

📄 License

This project is licensed under the MIT License.  
See the LICENSE file for details.

---

🤝 Contributing

Contributions are welcome.  
Please open an issue or submit a pull request.

---

🧭 Project Structure

`
tdgl-gl-benchmark/
│
├── tdgl_core/
│   ├── core.py
│   ├── observables.py
│   ├── utils.py
│   └── init.py
│
├── notebooks/
├── figures/
├── api.md
├── README.md
├── LICENSE
└── requirements.txt
`

---

📄 MIT License (LICENSE)

`text
MIT License

Copyright (c) 2025 ...

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
`

(Use the full MIT text in your repo.)

---

📘 api.md (Documentation Skeleton)

`markdown

API Documentation

tdgl_core.core

State
Dataclass containing ψ, A, μ fields.

Params
Dataclass containing geometry and physical parameters.

tdgl_step(state, params, dt)
One TDGL time step.

total_energy(state, params)
Compute GL free energy.

---

tdgl_core.observables

detectvorticesphase_winding(psi)
Phase‑winding vortex detector.

runHscan(H_vals)
Run TDGL relaxation over multiple external fields.

radial_profile(...)
Extract radial |ψ|² profile.

fitcoherencelength(...)
Fit GL analytic profile to extract ξ.

---

tdgl_core.utils

make_grid(...)
Generate coordinate grids.

set_seed(...)
Convenience PRNG helper.
`

