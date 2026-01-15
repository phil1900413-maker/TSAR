import numpy as np
import torch

from config import (
    ALL_POSSIBLE_GASES,
    NUM_CLASSES,
    TARGET_WN_START_CM,
    TARGET_WN_END_CM,
    MODEL_INPUT_POINTS,
    MIN_CONCENTRATIONS_ORDERED_LIST,
)
from data_utils import read_and_interpolate_experimental_spectrum
from model_hybrid import HybridCNNTransformer


DEFAULT_CONC_THRESHOLD_BY_GAS = {
    gas: float(lod) for gas, lod in zip(ALL_POSSIBLE_GASES, MIN_CONCENTRATIONS_ORDERED_LIST)
}


def _to_real(mu_log10):
    return np.power(10.0, mu_log10)


def _interval95_real(mu_log10, sigma_log10):
    lo = _to_real(mu_log10 - 1.96 * sigma_log10)
    hi = _to_real(mu_log10 + 1.96 * sigma_log10)
    return lo, hi


def smart_ensemble(mu_stack, var_stack, weights_by_gas):
    m, n, g = mu_stack.shape
    mu_out = np.zeros((n, g), dtype=np.float64)
    var_out = np.zeros((n, g), dtype=np.float64)
    gas_to_idx = {gas: i for i, gas in enumerate(ALL_POSSIBLE_GASES)}

    for gas, w in weights_by_gas.items():
        j = gas_to_idx[gas]
        w = np.asarray(w, dtype=np.float64)
        w = w / max(1e-12, w.sum())

        mu_j = mu_stack[:, :, j]
        var_j = var_stack[:, :, j]

        mu_hat = np.average(mu_j, axis=0, weights=w)
        mean_of_vars = np.average(var_j, axis=0, weights=w)
        var_of_means = np.average((mu_j - mu_hat) ** 2, axis=0, weights=w)

        mu_out[:, j] = mu_hat
        var_out[:, j] = mean_of_vars + var_of_means

    return mu_out, np.sqrt(np.maximum(var_out, 1e-18))


def predict_with_ensemble(models, x_batch, device, weights_by_gas):
    mu_list, sigma_list = [], []
    for model in models:
        out = model(x_batch)
        mu = out[:, :NUM_CLASSES]
        log_sigma = out[:, NUM_CLASSES:]
        mu_list.append(mu.detach().cpu().numpy())
        sigma_list.append(np.exp(log_sigma.detach().cpu().numpy()))
    mu_stack = np.stack(mu_list, axis=0)
    sigma_stack = np.stack(sigma_list, axis=0)
    var_stack = sigma_stack ** 2
    mu_s, sigma_s = smart_ensemble(mu_stack, var_stack, weights_by_gas)
    return mu_s, sigma_s


def detect_flags(mu_log10, sigma_log10, conc_thr_by_gas, sigma_thr_by_gas):
    conc = _to_real(mu_log10)
    flags = np.zeros_like(conc, dtype=bool)
    for j, gas in enumerate(ALL_POSSIBLE_GASES):
        c_thr = float(conc_thr_by_gas[gas])
        s_thr = float(sigma_thr_by_gas[gas])
        flags[:, j] = (conc[:, j] > c_thr) & (sigma_log10[:, j] < s_thr)
    return flags


def detection_metrics(flags, y_true_real, conc_thr_by_gas):
    lod = np.zeros((len(ALL_POSSIBLE_GASES),), dtype=np.float64)
    for j, gas in enumerate(ALL_POSSIBLE_GASES):
        lod[j] = float(conc_thr_by_gas[gas])

    y_pos = y_true_real > lod[None, :]
    out = {}

    for j, gas in enumerate(ALL_POSSIBLE_GASES):
        pred = flags[:, j]
        true = y_pos[:, j]

        tp = int(np.sum(pred & true))
        fp = int(np.sum(pred & (~true)))
        fn = int(np.sum((~pred) & true))
        tn = int(np.sum((~pred) & (~true)))

        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = 2 * prec * rec / max(1e-12, (prec + rec))
        fpr = fp / max(1, (fp + tn))

        out[gas] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr}

    return out


def fit_sigma_thresholds_min_fp(
    mu_log10,
    sigma_log10,
    y_true_real,
    conc_thr_by_gas=None,
    sigma_grid=None,
    max_fpr=0.001,
    recall_floor=0.0,
):
    if conc_thr_by_gas is None:
        conc_thr_by_gas = DEFAULT_CONC_THRESHOLD_BY_GAS
    if sigma_grid is None:
        sigma_grid = np.linspace(0.05, 1.5, 80, dtype=np.float64)

    best = {}
    lod = np.array([float(conc_thr_by_gas[g]) for g in ALL_POSSIBLE_GASES], dtype=np.float64)
    y_pos = y_true_real > lod[None, :]

    conc = _to_real(mu_log10)

    for j, gas in enumerate(ALL_POSSIBLE_GASES):
        best_t = float(sigma_grid[-1])
        best_fp = None
        best_rec = -1.0
        best_fpr = 1.0

        for t in sigma_grid:
            pred = (conc[:, j] > lod[j]) & (sigma_log10[:, j] < float(t))
            true = y_pos[:, j]

            tp = int(np.sum(pred & true))
            fp = int(np.sum(pred & (~true)))
            fn = int(np.sum((~pred) & true))
            tn = int(np.sum((~pred) & (~true)))

            rec = tp / max(1, (tp + fn))
            fpr = fp / max(1, (fp + tn))

            ok = (fpr <= float(max_fpr)) and (rec >= float(recall_floor))

            if ok:
                if best_fp is None:
                    best_fp = fp
                    best_fpr = fpr
                    best_rec = rec
                    best_t = float(t)
                else:
                    if fp < best_fp:
                        best_fp = fp
                        best_fpr = fpr
                        best_rec = rec
                        best_t = float(t)
                    elif fp == best_fp:
                        if rec > best_rec:
                            best_fpr = fpr
                            best_rec = rec
                            best_t = float(t)

        if best_fp is None:
            min_fp = None
            min_t = float(sigma_grid[-1])
            min_rec = -1.0
            min_fpr = 1.0
            for t in sigma_grid:
                pred = (conc[:, j] > lod[j]) & (sigma_log10[:, j] < float(t))
                true = y_pos[:, j]

                tp = int(np.sum(pred & true))
                fp = int(np.sum(pred & (~true)))
                fn = int(np.sum((~pred) & true))
                tn = int(np.sum((~pred) & (~true)))

                rec = tp / max(1, (tp + fn))
                fpr = fp / max(1, (fp + tn))

                if (min_fp is None) or (fp < min_fp) or (fp == min_fp and rec > min_rec):
                    min_fp = fp
                    min_t = float(t)
                    min_rec = rec
                    min_fpr = fpr

            best_t = float(min_t)
            best_fp = int(min_fp) if min_fp is not None else 0
            best_rec = float(min_rec)
            best_fpr = float(min_fpr)

        best[gas] = float(best_t)

    return best


@torch.no_grad()
def validate_real_files(
    file_paths,
    model_paths,
    weights_by_gas,
    sigma_thr_by_gas,
    conc_thr_by_gas=None,
    device="cuda",
):
    if conc_thr_by_gas is None:
        conc_thr_by_gas = DEFAULT_CONC_THRESHOLD_BY_GAS

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    models = []
    for p in model_paths:
        m = HybridCNNTransformer(mu_bias_init=None, log_sigma_bias_init=None).to(dev).eval()
        ckpt = torch.load(p, map_location=dev)
        sd = ckpt.get("model_state_dict", ckpt)
        fixed = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        m.load_state_dict(fixed)
        models.append(m)

    results = []
    for fp in file_paths:
        spec = read_and_interpolate_experimental_spectrum(
            fp, TARGET_WN_START_CM, TARGET_WN_END_CM, MODEL_INPUT_POINTS
        )
        if spec is None:
            continue

        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(dev)
        mu_s, sigma_s = predict_with_ensemble(models, x, dev, weights_by_gas)
        mu = mu_s[0]
        sigma = sigma_s[0]

        conc = _to_real(mu)
        lo, hi = _interval95_real(mu, sigma)

        verdict = []
        for j, gas in enumerate(ALL_POSSIBLE_GASES):
            if (conc[j] > float(conc_thr_by_gas[gas])) and (sigma[j] < float(sigma_thr_by_gas[gas])):
                verdict.append(gas)

        row = {"file": fp, "verdict": ",".join(verdict) if verdict else "-"}
        for j, gas in enumerate(ALL_POSSIBLE_GASES):
            row[f"{gas}_conc"] = float(conc[j])
            row[f"{gas}_lo95"] = float(lo[j])
            row[f"{gas}_hi95"] = float(hi[j])
            row[f"{gas}_sigma"] = float(sigma[j])
        results.append(row)

    return results
