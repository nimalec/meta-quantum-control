"""
PSD → Lindblad (physics-correct), with task distribution.

Assumptions:
- PSD S(ω; θ) is TWO-SIDED in angular frequency ω [rad/s], units [xi^2 * s].
- Coupling enters as energy: H_int = A * xi(t); pass A as g_energy_per_xi [J / xi].
  Examples:
    - Frequency noise δω(t) coupling as H_int = (ħ/2) δω σ_z  => g_energy_per_xi = ħ/2  [J·s · (rad/s)^-1]
    - Magnetic field noise B(t) with H_int = (g_e μ_B / 2) B σ_z => g_energy_per_xi = g_e μ_B / 2  [J/T]

Dephasing:
  γ_φ(T, sequence) = χ(T)/T with χ(T) = (1/π ħ²) ∫₀^∞ dω [ (gE)² S(ω)/ω² ] |F(ωT)|

Relaxation:
  Γ↓ = (gE)²/ħ² * S_eff(ω₀), where S_eff(ω₀) = S(ω₀) (Golden rule)
  or  S_eff = ∫ dω S(ω) L_Γ(ω-ω₀) for finite homogeneous linewidth Γ (normalized Lorentzian)

Outputs:
  Jump ops: √Γ↓ |g⟩⟨e|, √Γ↑ |e⟩⟨g|, √γ_φ σ_z/√2
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass

# ---------- constants ----------
TWO_PI = 2.0 * np.pi
HBAR = 1.054_571_817e-34  # J·s

# ---------- noise parameters ----------
@dataclass
class NoiseParameters:
    alpha: float   # spectral exponent for 1/f^alpha
    A: float       # amplitude/strength (units chosen so S has [xi^2 * s])
    omega_c: float # cutoff [rad/s]

    def to_array(self) -> np.ndarray:
        return np.array([self.alpha, self.A, self.omega_c], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "NoiseParameters":
        return cls(alpha=float(arr[0]), A=float(arr[1]), omega_c=float(arr[2]))

# ---------- PSD models ----------
class NoisePSDModel:
    """
    Power spectral density models S(ω; θ) with ω in rad/s.
    Returns TWO-SIDED PSD with units [xi^2 * s].
    Models:
      - 'one_over_f' : S(ω) = A / (|ω|^α + ω_c^α)
      - 'lorentzian' : S(ω) = A / (ω^2 + ω_c^2)         (OU noise; C(τ) ∝ e^{-ω_c |τ|})
      - 'double_exp' : sum of two Lorentzians (two time scales)
    """

    def __init__(self, model_type: str = "one_over_f"):
        self.model_type = model_type

    def psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        if self.model_type == "one_over_f":
            return self._one_over_f_psd(omega, theta)
        elif self.model_type == "lorentzian":
            return self._lorentzian_psd(omega, theta)
        elif self.model_type == "double_exp":
            return self._double_exp_psd(omega, theta)
        else:
            raise ValueError(f"Unknown model_type='{self.model_type}'")

    def _one_over_f_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        # S(ω) = A / (|ω|^α + ω_c^α)
        eps = 1e-30
        if theta.alpha > 0:
            denom = np.power(np.abs(omega), theta.alpha) + (theta.omega_c ** theta.alpha)
        else:
            denom = np.ones_like(omega)
        return theta.A / (denom + eps)

    def _lorentzian_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        # S(ω) = A / (ω² + ω_c²)
        eps = 1e-30
        return theta.A / (omega**2 + theta.omega_c**2 + eps)

    def _double_exp_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        # Two Lorentzians with coarse interpolation via alpha
        eps = 1e-30
        omega_c1 = theta.omega_c
        omega_c2 = theta.omega_c * (1.0 + max(theta.alpha, 0.0))
        A1 = theta.A * (1 - min(max(theta.alpha, 0.0), 5.0) / 5.0)
        A2 = theta.A - A1
        return A1 / (omega**2 + omega_c1**2 + eps) + A2 / (omega**2 + omega_c2**2 + eps)

    def correlation_function(self, tau: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """C(τ) via inverse FT (approx). For Lorentzian: C(τ)= (A/2ω_c) e^{-ω_c|τ|}."""
        tau = np.asarray(tau, dtype=float)
        if self.model_type == "lorentzian":
            return (theta.A / (2.0 * theta.omega_c)) * np.exp(-theta.omega_c * np.abs(tau))
        # Numeric inverse FT for other models (coarse)
        w = np.linspace(-1e3*theta.omega_c - 1.0, 1e3*theta.omega_c + 1.0, 20001)
        S_w = self.psd(w, theta)
        C_tau = np.trapz(S_w[:, None] * np.exp(1j * w[:, None] * tau), w, axis=0) / (2.0 * np.pi)
        return C_tau.real

# ---------- PSD → Lindblad ----------
class PSDToLindblad:
    """
    Map a colored PSD to Lindblad jump operators (qubit).
    - Dephasing (σ_z):   γ_φ(T, sequence) from filter-function integral (χ(T)/T)
    - Relaxation (σ±):   Γ↓, Γ↑ from PSD at ±ω₀ (or convolved with line shape if Γ_h>0)

    Inputs:
      psd_model: returns two-sided angular PSD S(ω) with units [xi^2 * s]
      g_energy_per_xi: coupling coefficient so that (g^2 S)/ħ^2 has units 1/s
                       e.g., frequency noise δω with H_int = (ħ/2) δω σ_z → g = ħ/2
    """

    def __init__(self, psd_model: "NoisePSDModel", g_energy_per_xi: float, hbar: float = 1.054_571_817e-34):
        self.psd_model = psd_model
        self.gE = float(g_energy_per_xi)  # [J / xi]
        self.hbar = float(hbar)

    # ----- filter functions for dephasing -----
    @staticmethod
    def _F_abs(omega_T: np.ndarray, sequence: str) -> np.ndarray:
        seq = sequence.lower()
        if seq == "ramsey":
            return 4.0 * np.sin(0.5 * omega_T) ** 2
        if seq == "echo":
            return 8.0 * np.sin(0.25 * omega_T) ** 4
        if seq.startswith("cpmg_"):
            n = int(seq.split("_")[1])
            num = np.sin(n * omega_T / 2.0)
            den = np.maximum(np.sin(omega_T / 2.0), 1e-30)
            return 8.0 * np.sin(omega_T / (4.0 * n)) ** 4 * (num / den) ** 2
        raise ValueError(f"Unknown sequence '{sequence}'")

    def dephasing_rate(self, theta: "NoiseParameters", T: float, sequence: str = "ramsey",
                       omega_max_factor: float = 20.0, n_w: int = 4000) -> float:
        """
        γ_φ = χ(T)/T, χ(T) = (1/π ħ²) ∫₀^∞ dω [ (gE)² S(ω;θ) / ω² ] |F(ωT)|.
        """
        T = max(float(T), 1e-15)
        w_max = omega_max_factor / T
        w = np.linspace(0.0, w_max, int(n_w))
        S_w = self.psd_model.psd(w, theta)                                # [xi^2 * s]
        F = self._F_abs(w * T, sequence)
        integrand = (self.gE ** 2) * S_w * (F / np.maximum(w, 1e-30) ** 2)  # J^2 * s
        chi = (1.0 / np.pi) * np.trapz(integrand, w) / (self.hbar ** 2)     # dimensionless
        gamma_phi = max(chi / T, 0.0)                                        # 1/s
        return gamma_phi

    # ----- relaxation (σ±) -----
    @staticmethod
    def _lorentzian_L(w: np.ndarray, w0: float, Gamma_h: float) -> np.ndarray:
        """Normalized Lorentzian line shape centered at w0, FWHM=Gamma_h."""
        return (Gamma_h / 2.0) / np.pi / ((w - w0) ** 2 + (Gamma_h / 2.0) ** 2)

    def relaxation_rates(self, theta: "NoiseParameters", omega0: float,
                         temperature_K: float | None = None,
                         Gamma_h: float = 0.0,
                         span_factor: float = 50.0,
                         n_w: int = 4001) -> Tuple[float, float]:
        """
        Γ↓, Γ↑ from S_eff at ω0:
          - If Gamma_h == 0:  S_eff = S(ω0)
          - Else:             S_eff = ∫ S(ω) L_Γ(ω-ω0) dω   (L_Γ normalized)
        Γ↑ = e^{-β ħ ω0} Γ↓ if temperature_K provided; else Γ↑=Γ↓ (classical PSD).
        """
        pref = (self.gE ** 2) / (self.hbar ** 2)  # 1/s per [xi^2 * s]
        if Gamma_h > 0.0:
            w_span = max(span_factor * max(Gamma_h, 1e-12), 5.0 * abs(omega0))
            w = np.linspace(omega0 - w_span, omega0 + w_span, int(n_w))
            L = self._lorentzian_L(w, omega0, Gamma_h)
            S_w = self.psd_model.psd(w, theta)
            S_eff = np.trapz(S_w * L, w)  # [xi^2 * s]
        else:
            S_eff = float(self.psd_model.psd(np.array([abs(omega0)]), theta)[0])

        Gamma_down = max(pref * S_eff, 0.0)

        if temperature_K is None:
            Gamma_up = Gamma_down
        else:
            import scipy.constants as sc
            beta = 1.0 / (sc.k * float(temperature_K))
            Gamma_up = max(np.exp(-beta * self.hbar * abs(omega0)) * Gamma_down, 0.0)

        return Gamma_down, Gamma_up

    # ----- assemble qubit jump operators -----
    def qubit_lindblad_ops(self, theta: "NoiseParameters", *,
                           T: float, sequence: str,
                           omega0: float,
                           temperature_K: float | None = None,
                           Gamma_h: float = 0.0):
        """
        Returns:
            ops:  [ √Γ↓ |g><e|, √Γ↑ |e><g|, √γ_φ (σ_z/√2) ]
            rates: np.array([Γ↓, Γ↑, γ_φ])  [1/s]
        """
        # dephasing
        gamma_phi = self.dephasing_rate(theta, T=T, sequence=sequence)
        L_phi = (1.0 / np.sqrt(2.0)) * np.array([[1, 0], [0, -1]], dtype=complex)

        # relaxation
        Gamma_down, Gamma_up = self.relaxation_rates(theta, omega0, temperature_K, Gamma_h)
        L_minus = np.array([[0, 0], [1, 0]], dtype=complex)  # |g><e|
        L_plus  = np.array([[0, 1], [0, 0]], dtype=complex)  # |e><g|

        ops = [np.sqrt(Gamma_down) * L_minus,
               np.sqrt(gamma_phi)  * L_phi]
        rates = np.array([Gamma_down, Gamma_up, gamma_phi], dtype=float)
        return ops, rates

    def get_effective_rates(self, theta: "NoiseParameters", *,
                            T: float, sequence: str,
                            omega0: float,
                            temperature_K: float | None = None,
                            Gamma_h: float = 0.0) -> Dict[str, float]:
        """Convenience: return {'Gamma_down','Gamma_up','gamma_phi'} in 1/s."""
        gamma_phi = self.dephasing_rate(theta, T=T, sequence=sequence)
        Gamma_down, Gamma_up = self.relaxation_rates(theta, omega0, temperature_K, Gamma_h)
        return {"Gamma_down": Gamma_down, "Gamma_up": Gamma_up, "gamma_phi": gamma_phi}


# ---------- task distribution ----------
class TaskDistribution:
    """
    Distribution P over θ = (alpha, A, omega_c).
    Provides 'uniform' and 'gaussian' sampling.
    """

    def __init__(self,
                 dist_type: str = "uniform",
                 ranges: Dict[str, Tuple[float, float]] | None = None,
                 mean: np.ndarray | None = None,
                 cov: np.ndarray | None = None):
        self.dist_type = dist_type
        self.ranges = ranges or {
            "alpha":   (0.5, 2.0),
            "A":       (0.01, 0.5),
            "omega_c": (1.0, 10.0),
        }
        self.mean = mean
        self.cov = cov

    def sample(self, n_tasks: int, rng: np.random.Generator | None = None) -> List[NoiseParameters]:
        rng = rng or np.random.default_rng()
        if self.dist_type == "uniform":
            return self._sample_uniform(n_tasks, rng)
        elif self.dist_type == "gaussian":
            return self._sample_gaussian(n_tasks, rng)
        else:
            raise ValueError(f"Unknown distribution type '{self.dist_type}'")

    def _sample_uniform(self, n: int, rng: np.random.Generator) -> List[NoiseParameters]:
        tasks: List[NoiseParameters] = []
        for _ in range(n):
            alpha   = rng.uniform(*self.ranges["alpha"])
            A       = rng.uniform(*self.ranges["A"])
            omega_c = rng.uniform(*self.ranges["omega_c"])
            tasks.append(NoiseParameters(alpha=float(alpha), A=float(A), omega_c=float(omega_c)))
        return tasks

    def _sample_gaussian(self, n: int, rng: np.random.Generator) -> List[NoiseParameters]:
        if self.mean is None or self.cov is None:
            raise ValueError("Gaussian sampling requires 'mean' and 'cov'")
        samples = rng.multivariate_normal(self.mean, self.cov, size=n)
        return [NoiseParameters.from_array(s) for s in samples]

    def compute_variance(self) -> float:
        if self.dist_type == "uniform":
            var_alpha = ((self.ranges["alpha"][1]   - self.ranges["alpha"][0])   ** 2) / 12.0
            var_A     = ((self.ranges["A"][1]       - self.ranges["A"][0])       ** 2) / 12.0
            var_omega = ((self.ranges["omega_c"][1] - self.ranges["omega_c"][0]) ** 2) / 12.0
            return float(var_alpha + var_A + var_omega)
        elif self.dist_type == "gaussian":
           # return 0.0
            return float(np.trace(self.cov))

# ---------- utilities ----------
def psd_distance(psd_model: NoisePSDModel,
                 theta1: NoiseParameters,
                 theta2: NoiseParameters,
                 omega_grid: np.ndarray) -> float:
    """d(θ,θ') = sup_ω |S(ω;θ) - S(ω;θ')| over provided grid"""
    S1 = psd_model.psd(omega_grid, theta1)
    S2 = psd_model.psd(omega_grid, theta2)
    return float(np.max(np.abs(S1 - S2)))

# ---------- example usage ----------
if __name__ == "__main__":
    # 1) Task distribution over PSD params (keep your ranges)
    task_dist = TaskDistribution(
        dist_type="uniform",
        ranges={"alpha": (0.1, 4.0), "A": (100, 1e5), "omega_c": (0, 800)}
    )
    rng = np.random.default_rng()
    tasks = task_dist.sample(10, rng)

    print("Sampled tasks:")
    for i, t in enumerate(tasks):
        print(f"  Task {i}: α={t.alpha:.2f}, A={t.A:.3f}, ωc={t.omega_c:.2f} rad/s")

    # 2) Pick a PSD model (no NV specifics)
    psd_model = NoisePSDModel(model_type="one_over_f")

    # 3) Converter: choose coupling scale.
    # Example: frequency noise δω(t) with H_int = (ħ/2) δω σ_z  ⇒ gE = ħ/2
    converter = PSDToLindblad(psd_model, g_energy_per_xi=HBAR/2)

    # 4) Set qubit transition and experiment window (generic)
    omega0 = 2 * np.pi    # 5 MHz (rad/s)
    T = 1                  # 50 μs window
    sequence = "ramsey"        # 'ramsey' | 'echo' | 'cpmg_n'
    temperature_K = None       # classical PSD ⇒ Γ↑=Γ↓
    Gamma_h = 0.0              # set >0 to include homogeneous broadening (rad/s)

    # 5) Build jump operators/rates for each sampled task
    for i, t in enumerate(tasks):
        ops, rates = converter.qubit_lindblad_ops(
            t, T=T, sequence=sequence, omega0=omega0, temperature_K=temperature_K, Gamma_h=Gamma_h
        )
        print(f"\nTask {i} rates [1/s]: Γ↓={rates[0]:.3e}, Γ↑={rates[1]:.3e}, γφ={rates[2]:.3e}")

    # 6) (Optional) visualize PSDs
    try:
        import matplotlib.pyplot as plt
        omega = np.logspace(-1, 3, 1000)
        plt.figure(figsize=(9, 5))
        for i, t in enumerate(tasks):
            S = psd_model.psd(omega, t)
            plt.loglog(omega, S, label=f'Task {i}')
        plt.xlabel('ω (rad/s)')
        plt.ylabel('S(ω)  [xi²·s]')
        plt.title('Sampled PSDs (two-sided, angular)')
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig("psd_samples.png", dpi=150)
        print("\nSaved PSD plot → psd_samples.png")
    except Exception as e:
        print(f"(Plot skipped) {e}")

    # 7) Pairwise PSD distances
    print("\nPairwise PSD sup distances:")
    omega_grid = np.logspace(-1, 2, 500)
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            d = psd_distance(psd_model, tasks[i], tasks[j], omega_grid)
            print(f"  d(task{i}, task{j}) = {d:.4e}")

    # 8) Variance of task distribution (for your theory bounds)
    print(f"\nσ²_θ (sum of per-dim variances) = {task_dist.compute_variance():.4f}")
