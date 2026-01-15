from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np

import saturation_effects


THZ_TO_CM_INV = 33.3564095198152


@dataclass(frozen=True)
class GeneratorConfig:
    wn0_cm: float
    wn1_cm: float
    min_step: float
    max_step: float

    log_min_conc: float
    log_max_conc: float

    min_gauss_sigma: float
    max_gauss_sigma: float
    prob_gauss: float

    min_baseline_offset: float
    max_baseline_offset: float
    prob_offset: float

    drift_amp_min: float
    drift_amp_max: float

    max_observable_intensity: float = 100.0


@dataclass(frozen=True)
class Profiles:
    baseline_x: Optional[np.ndarray] = None
    baseline_mean: Optional[np.ndarray] = None
    baseline_delta: Optional[np.ndarray] = None

    noise_x: Optional[np.ndarray] = None
    noise_mean: Optional[np.ndarray] = None

    pca_x: Optional[np.ndarray] = None
    pca_components: Optional[np.ndarray] = None
    pca_variances: Optional[np.ndarray] = None

    cutoff_x_cm: Optional[np.ndarray] = None
    cutoff_y: Optional[np.ndarray] = None


def make_grid(cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    dx = float(rng.uniform(cfg.min_step, cfg.max_step))
    n = int(round((cfg.wn1_cm - cfg.wn0_cm) / dx)) + 1
    return np.linspace(cfg.wn0_cm, cfg.wn1_cm, n, dtype=np.float64)


def interp_spectrum(x_grid: np.ndarray, x_src: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    idx = np.argsort(x_src)
    return np.interp(x_grid, x_src[idx], y_src[idx], left=0.0, right=0.0).astype(np.float64)


def sample_concentration(cfg: GeneratorConfig, rng: np.random.Generator) -> float:
    return float(10 ** rng.uniform(cfg.log_min_conc, cfg.log_max_conc))


def pca_drift(x: np.ndarray, prof: Profiles, cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    if prof.baseline_x is None or prof.baseline_mean is None or prof.baseline_delta is None:
        return np.zeros_like(x)

    base_mean = np.interp(x, prof.baseline_x, prof.baseline_mean).astype(np.float64)

    if prof.pca_x is None or prof.pca_components is None:
        delta = np.mean(np.abs(np.interp(x, prof.baseline_x, prof.baseline_delta)))
        x_norm = (x - x[0]) / max(1e-12, (x[-1] - x[0])) * 2.0 - 1.0
        a = float(rng.uniform(-delta, delta))
        b = float(rng.uniform(-1.5 * delta, 1.5 * delta))
        c = float(rng.uniform(-delta, delta))
        return base_mean + (a * x_norm * x_norm + b * x_norm + c)

    comps = np.empty((x.size, prof.pca_components.shape[1]), dtype=np.float64)
    for i in range(prof.pca_components.shape[1]):
        comps[:, i] = np.interp(x, prof.pca_x, prof.pca_components[:, i])

    if prof.pca_variances is not None and prof.pca_variances.size == comps.shape[1]:
        weights = np.sqrt(prof.pca_variances / max(1e-12, prof.pca_variances[0]))
    else:
        weights = np.ones(comps.shape[1], dtype=np.float64)

    coeffs = rng.normal(loc=0.0, scale=weights, size=comps.shape[1]).astype(np.float64)
    drift = comps @ coeffs
    drift_std = float(np.std(drift))
    if drift_std > 1e-12:
        drift *= float(np.mean(prof.baseline_delta)) / drift_std

    amp = float(rng.uniform(cfg.drift_amp_min, cfg.drift_amp_max))
    return base_mean + drift * amp


def jagged_anchors(x0: float, x1: float, dx: float, rng: np.random.Generator,
                   min_mult: float = 8.0, max_mult: float = 20.0) -> np.ndarray:
    pts = [x0]
    cur = x0
    while cur < x1:
        step = float(dx * rng.uniform(min_mult, max_mult))
        step = step if step > 1e-12 else dx * min_mult
        cur = min(x1, cur + step)
        pts.append(cur)
    return np.array(pts, dtype=np.float64)


def apply_noise(y: np.ndarray, x: np.ndarray, prof: Profiles, cfg: GeneratorConfig,
                rng: np.random.Generator, dx: float) -> Tuple[np.ndarray, bool]:
    y_out = y.copy().astype(np.float64)

    if prof.noise_x is not None and prof.noise_mean is not None and prof.noise_x.size >= 2:
        base = np.interp(x, prof.noise_x, prof.noise_mean, left=0.0, right=prof.noise_mean[-1]).astype(np.float64)
        base *= float(rng.uniform(0.9, 1.4))

        ax = jagged_anchors(float(x[0]), float(x[-1]), dx, rng)
        ay = np.interp(ax, x, base).astype(np.float64)
        jagged = np.interp(x, ax, ay).astype(np.float64)

        y_out += jagged
        sigma = np.abs(base) * 1.0
        y_out += rng.normal(loc=0.0, scale=sigma, size=y_out.shape).astype(np.float64)

        return y_out, True

    if float(rng.random()) < cfg.prob_gauss:
        sigma = float(rng.uniform(cfg.min_gauss_sigma, cfg.max_gauss_sigma))
        y_out += rng.normal(loc=0.0, scale=sigma, size=y_out.shape).astype(np.float64)

    return y_out, False


def apply_offsets(y: np.ndarray, cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    y_out = y.copy().astype(np.float64)
    if float(rng.random()) < cfg.prob_offset:
        y_out += float(rng.uniform(cfg.min_baseline_offset, cfg.max_baseline_offset))
    return y_out


def dynamic_cutoff(x: np.ndarray, prof: Profiles, cfg: GeneratorConfig, rng: np.random.Generator,
                   drift_component: np.ndarray) -> Optional[np.ndarray]:
    if prof.cutoff_x_cm is None or prof.cutoff_y is None or prof.cutoff_x_cm.size < 2:
        return None

    base_cut = np.interp(x, prof.cutoff_x_cm, prof.cutoff_y).astype(np.float64)

    lower = base_cut * 0.8
    upper = np.minimum(base_cut * 10.0, float(cfg.max_observable_intensity))

    n_anchors = int(rng.integers(15, 30))
    ax = np.linspace(float(x[0]), float(x[-1]), n_anchors, dtype=np.float64)
    ay = np.zeros_like(ax)

    for i, xv in enumerate(ax):
        lo = float(np.interp(xv, x, lower))
        hi = float(np.interp(xv, x, upper))
        ay[i] = float(rng.uniform(lo, hi)) if hi > lo else lo

    jerky = np.interp(x, ax, ay).astype(np.float64)

    coupling = float(rng.uniform(0.2, 1.0))
    return jerky + drift_component * coupling


def apply_realistic_saturation(y: np.ndarray, y_cutoff: Optional[np.ndarray], x: np.ndarray,
                               rng: np.random.Generator) -> np.ndarray:
    if y_cutoff is None:
        return np.clip(y, a_min=None, a_max=float(np.max(y)))

    params = {
        "jaggedness": float(rng.uniform(0.05, 0.14)),
        "width_threshold": int(rng.integers(10, 400)),
        "crater_min_depth": float(rng.uniform(-0.20, -0.08)),
        "crater_max_bulge": float(rng.uniform(0.08, 0.20)),
        "tower_height_range": (0.8, 1.5),
        "tower_width_pts": 4,
        "moat_depth_factor": 0.85,
        "moat_noise_factor": 0.05,
    }
    return saturation_effects.apply_smart_saturation(y, y_cutoff, x, params)


def generate_one_mixture(
    gas_library: Dict[str, Tuple[np.ndarray, np.ndarray]],
    active_gases: List[str],
    prof: Profiles,
    cfg: GeneratorConfig,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))

    x = make_grid(cfg, rng)
    dx = float((x[-1] - x[0]) / max(1, x.size - 1))

    y = np.zeros_like(x, dtype=np.float64)
    conc = {}

    for g in active_gases:
        xg, yg = gas_library[g]
        c = sample_concentration(cfg, rng)
        y += interp_spectrum(x, xg, yg) * c
        conc[g] = c

    bl_plus_drift = pca_drift(x, prof, cfg, rng)
    drift_component = bl_plus_drift - np.interp(x, prof.baseline_x, prof.baseline_mean).astype(np.float64) if prof.baseline_x is not None and prof.baseline_mean is not None else np.zeros_like(x)

    y = y + bl_plus_drift
    y, _ = apply_noise(y, x, prof, cfg, rng, dx)
    y = apply_offsets(y, cfg, rng)

    y_cut = dynamic_cutoff(x, prof, cfg, rng, drift_component)
    y = apply_realistic_saturation(y, y_cut, x, rng)

    y = np.clip(y, a_min=None, a_max=float(cfg.max_observable_intensity))
    return x, y
