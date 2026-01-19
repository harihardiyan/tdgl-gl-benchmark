
# TDGL Ginzburg–Landau Benchmark Suite (JAX)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/JAX-Accelerated-red?logo=google&logoColor=white" alt="JAX">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Hardware-GPU%20%2F%20TPU-orange" alt="Hardware">
</p>

A modular, reproducible, and research‑grade implementation of the **Time‑Dependent Ginzburg–Landau (TDGL)** equations for 2D superconductors. Built using **JAX** for high‑performance automatic differentiation, JIT compilation, and seamless GPU/TPU acceleration.

---

## 📌 Quick Navigation
- [Interactive Notebooks](#-interactive-notebooks-google-colab)
- [Key Features](#-key-features)
- [Scientific Background](#-scientific-background)
- [Installation](#-installation)
- [Usage Example](#-usage-example)
- [Documentation](#-documentation)
- [Academic Integrity](#-what-this-project-does-not-claim)

---

## 🧪 Interactive Notebooks (Google Colab)

Run these 7 benchmark simulations directly in your browser. Each notebook covers a specific aspect of the TDGL model:

### 1. Single Vortex Dynamics
Fundamental simulation focusing on the formation and stability of a single vortex core.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harihardiyan/tdgl-gl-benchmark/blob/main/notebooks/01_single_vortex.ipynb)

### 2. Multi-Vortex State (64×64)
Observation of vortex lattice formation and equilibrium states on a standard resolution grid.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harihardiyan/tdgl-gl-benchmark/blob/main/notebooks/Multi%E2%80%91Vortex_State_(64%C3%9764).ipynb)

### 3. Multi-Vortex State (128×128)
High-resolution simulation for detailed visualization of vortex distribution and interaction patterns.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harihardiyan/tdgl-gl-benchmark/blob/main/notebooks/Multi%E2%80%91Vortex_128%C3%97128.ipynb)

### 4. H-Scan Observables
Calculation of physical observables and phase transitions across a range of external magnetic fields ($H$).
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harihardiyan/tdgl-gl-benchmark/blob/main/notebooks/H%E2%80%91Scan_Observables.ipynb)

### 5. Phase Diagram (H vs T)
Mapping the superconducting-to-normal phase boundary within the magnetic field ($H$) and temperature ($T$) plane.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harihardiyan/tdgl-gl-benchmark/blob/main/notebooks/Phase_Diagram_(H_vs_T).ipynb)

### 6. Differentiable Inverse Design
Advanced application using JAX gradients to optimize the magnetic field ($B_0$) for a specific target vortex count.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harihardiyan/tdgl-gl-benchmark/blob/main/notebooks/Differentiable_Inverse_Design_(Optimize_B0_for_Target_Vortex_Count).ipynb)

### 7. Vortex Drift & Diode Effect
Simulation of vortex motion driven by bias current and investigation of the superconducting diode effect.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harihardiyan/tdgl-gl-benchmark/blob/main/notebooks/vortex_drift___diode_effect__tdgl___bias_current.ipynb)

---

## ✨ Key Features

### TDGL Core
- **PyTree‑based State and Params:** Clean JAX integration for optimization and autodiff.
- **Gauge‑invariant:** Link‑variable covariant gradient implementation.
- **Physics-rich:** Full GL energy functional and $\mu$-Poisson solver (SOR) for gauge fixing.
- **Performance:** JIT‑accelerated time stepping with uniform geometry initialization.

### Observables Suite
- **Vortex Detection:** High-accuracy detection via phase winding.
- **Radial Profiles:** Extraction of vortex profiles and fitting for coherence length $\xi$.
- **Magnetic Analysis:** $H$-field scans (Meissner → Vortex → High‑field).
- **Statistical Tools:** Vortex count, density, and variance indicators.
- **Critical Fields:** Numerical estimates of $H_{c1}$ and $H_{c2}$.

---

## 🧠 Scientific Background

The TDGL equations describe the dynamics of the superconducting order parameter $\psi(\mathbf{r}, t)$ and vector potential $\mathbf{A}(\mathbf{r}, t)$:

$$
\frac{\partial \psi}{\partial t} = -\gamma_\psi \frac{\delta F}{\delta \psi^*}, \qquad \frac{\partial \mathbf{A}}{\partial t} = -\gamma_A \left( \frac{\delta F}{\delta \mathbf{A}} + \mathbf{J}_n \right)
$$

The **Ginzburg–Landau free energy density** is given by:

$$
f = a|\psi|^2 + \frac{1}{2}|\psi|^4 + |(\nabla - i\mathbf{A})\psi|^2 + \kappa^2 (B - H_\text{ext})^2
$$

Where the normal current $\mathbf{J}_n = -\sigma_n \nabla \mu$. This repository implements these using gauge‑invariant link variables and Neumann boundary conditions.

---

## 🚫 Academic Disclaimer

To maintain academic integrity, this project does **not** claim:
- To be a full microscopic model (e.g., BdG, Eilenberger, or Usadel equations).
- To simulate quantum fluctuations or microscopic pairing mechanisms.
- To provide production‑grade device optimization without experimental calibration.
- To implement 3D TDGL or anisotropic materials in the current version.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/harihardiyan/tdgl-gl-benchmark.git
cd tdgl-gl-benchmark

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Usage Example

```python
from tdgl_core import (
    initparamsbasic,
    initstaterandom,
    tdgl_step,
    total_energy,
    detectvorticesphase_winding
)

# Initialize parameters and state
params = initparamsbasic(nx=64, ny=64, B0=0.1)
state = initstaterandom(params)

# Simulation loop
for _ in range(1000):
    state = tdgl_step(state, params, dt=0.01)

# Analyze results
psi = state.psi
vortices = detectvorticesphase_winding(psi)
energy = total_energy(state, params)
print(f"Energy: {energy:.4f} | Vortex count: {len(vortices)}")
```

---

## 🧭 Project Structure

```text
tdgl-gl-benchmark/
├── tdgl_core/          # Core solver and observables logic
│   ├── core.py         # TDGL time-stepping & Poisson solver
│   ├── observables.py  # Vortex detection & physical metrics
│   └── utils.py        # Grid generation & helper functions
├── notebooks/          # Google Colab compatible examples
├── figures/            # Generated simulation plots
├── api.md              # Detailed API Documentation
└── requirements.txt    # Project dependencies
```

---

## 📊 Citation

If you use this repository in academic work, please cite:

```text
Hari, et al. (2025). 
TDGL Ginzburg–Landau Benchmark Suite (JAX). 
GitHub: https://github.com/harihardiyan/tdgl-gl-benchmark
```

---

## 📄 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
```

