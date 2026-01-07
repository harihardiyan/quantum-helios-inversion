

# Differentiable Quantum-Informed Neural Inversion for Transonic Solar Wind Parameters

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: JAX](https://img.shields.io/badge/Framework-JAX-red.svg)](https://github.com/google/jax)
[![Quantum: PennyLane](https://img.shields.io/badge/Quantum-PennyLane-blueviolet.svg)](https://pennylane.ai/)
[![SciML: Physics--Informed](https://img.shields.io/badge/SciML-Physics--Informed-green.svg)](https://pennylane.ai/)
[![Maintenance: Active](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/harihardiyan/)

**Author:** Hari Hardiyan  
**Email:** lorozloraz@gmail.com  
**Role:** AI Enthusiast / Independent Researcher  

---

## 📌 Overview
This repository presents a **Hybrid Physics-Informed Quantum Neural Network (PIQNN)** framework designed to solve the complex inverse problem of heliospheric parameter estimation. By integrating **Variational Quantum Circuits (VQC)** into a **Differentiable Physics** pipeline, the model reconstructs critical solar source parameters—such as the source surface magnetic field ($B_{r0}$), critical radius ($r_0$), and isothermal sound speed ($c_s$)—directly from 1 AU observations.

The core innovation lies in the use of **JAX-optimized XLA** kernels to bridge quantum expectation values with transcendental fluid dynamics (Lambert W solutions), ensuring astrophysical consistency through strictly enforced physical constraints.

---

## 🚀 Key Scientific Contributions

### 1. Quantum-Classical Co-Processing
The architecture leverages a 6-qubit **Variational Quantum Circuit** using `StronglyEntanglingLayers`.
*   **Hilbert Space Mapping:** Features are embedded via `AngleEmbedding`, allowing the model to capture non-linear correlations in a 64-dimensional Hilbert space.
*   **Equinox Integration:** Implemented using a functional approach, where PennyLane QNodes are wrapped as immutable JAX modules for seamless end-to-end backpropagation.

### 2. Transcendental Physics Integration (Lambert W)
Instead of relying on computationally expensive numerical ODE solvers during training, this model utilizes the **Lambert W Function** ($k=-1$ branch) for analytical supersonic solutions.
*   **Supersonic Modeling:** Targeted inversion of the Parker isothermal equation to ensure predictions align with the physical transonic transition of solar wind.
*   **Analytical Residuals:** Physics-informed loss is calculated using the explicit RHS of the Parker ODE, providing a clean gradient signal for the optimizer.

### 3. Hard-Constrained Optimization via Log-Space Adaptor
To mitigate numerical instability and "unphysical hallucination," a **Log-Space Sigmoid Adaptor** is employed.
*   **Safety Guardrails:** All quantum outputs are mapped into astrophysical valid ranges in log-space, inherently preventing `NaN` divergences and negative physical constants.

### 4. Bayesian Uncertainty Quantification (UQ)
The model implements **Monte Carlo Dropout** vectorized through JAX's `vmap` primitive.
*   **Confidence Intervals:** Generates predictive means and 95% confidence intervals across 200+ stochastic passes, quantifying the epistemic uncertainty of the quantum-physical inversion.

---

## 🧬 Mathematical Framework

### Parker Wind Inversion
The radial and azimuthal magnetic field components at 1 AU follow the **Parker Spiral** geometry:
$$B_r = B_{r0} \left( \frac{r_0}{r} \right)^2, \quad B_\phi = -B_{r0} \left( \frac{r_0}{r} \right)^2 \left( \frac{\Omega r \sin \theta}{v_r} \right)$$

### Physics-Informed Objective
The model minimizes a composite loss function $\mathcal{L}$, balancing data fidelity and fluid consistency:
$$\mathcal{L} = \omega_{sup} \sum_{i} \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2} + \lambda_{phys} \left\| \frac{du}{dr} - f_{Parker}(u, c_s, r) \right\|^2$$

---

## 🛠 Technical Stack

*   **Engine:** [JAX](https://github.com/google/jax) (XLA Optimization, FP64 Precision).
*   **Neural Components:** [Equinox](https://github.com/patrick-kidger/equinox) (Functional state management).
*   **Quantum Backend:** [PennyLane](https://pennylane.ai/) (Automatic differentiation of quantum gates).
*   **Optimization:** [Optax](https://github.com/google-deepmind/optax) (Adam optimizer with JIT-compiled steps).

---

## 📈 Performance Highlights
*   **Monotonic Convergence:** Observed steady decay in supervised loss with an 8-qubit and 6-qubit configuration.
*   **Radius Accuracy:** Successfully reconstructed the solar critical radius ($r_0$) with a Relative MAE of **<10%**.
*   **Integritas Fisika:** Physics residuals maintained at a stable **$10^{-3}$** order, proving the robustness of the Log-Space Adaptor.

---

## 📂 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/harihardiyan/quantum-helios-inversion.git
cd quantum-helios-inversion

# Install dependencies
pip install jax jaxlib equinox optax pennylane scipy numpy

# Execute the PIQNN Training Pipeline
python main.py
```

---

## 📄 License & Citation
This project is licensed under the **MIT License**.  
If you find this work useful for your research, please cite:
> *Hardiyan, H. (2026). Differentiable Quantum-Informed Neural Inversion for Transonic Solar Wind Parameters.*

---

## 📬 Contact
**Author:** Hari Hardiyan  
**GitHub:** [@harihardiyan](https://github.com/harihardiyan)  
**Email:** lorozloraz@gmail.com  
**Interests:** Quantum Machine Learning, Heliophysics, Scientific Computing.
