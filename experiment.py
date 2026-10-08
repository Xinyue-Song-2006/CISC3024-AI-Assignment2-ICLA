"""AI-written ICLA digit-recognition experiment (CISC3024 Assignment 2).

Run: python experiment.py --output results
ICLA reference: Zhdanov et al., WACV 2025, arXiv:2312.10464.
Independent NumPy implementation; no upstream source is copied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

METHODS = ("MAP", "LLLA-EF", "ICLA-EB", "ICLA-Val")
SEEDS = (0, 1, 2, 3, 4)


def features(model, x):
    hidden = np.maximum(0.0, x @ model.coefs_[0] + model.intercepts_[0])
    return np.column_stack((hidden, np.ones(len(x))))


def last_weights(model):
    return np.vstack((model.coefs_[1], model.intercepts_[1]))


def ef_diagonal(phi, probabilities, y):
    residual = probabilities - np.eye(probabilities.shape[1])[y]
    return (phi * phi).T @ (residual * residual)


def evidence_precision(h, weights):
    """Frozen-mode diagonal Laplace evidence optimization in log precision.

    Up to lambda-independent terms, negative log evidence equals
    0.5 * [lambda * ||w||^2 + sum(log(1 + h_j/lambda))].
    This is an empirical-Bayes approximation; the mode is NOT retrained.
    """
    norm2 = np.square(weights).sum()

    def objective(log_precision):
        precision = np.exp(log_precision)
        return 0.5 * (precision * norm2 + np.log1p(h / precision).sum())

    result = minimize_scalar(objective, bounds=(-12, 12), method="bounded",
                             options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError("Prior precision optimization failed")
    return float(np.exp(result.x))


def predict(phi, weights, method, precision=None, h=None):
    logits = phi @ weights
    if method == "MAP":
        return softmax(logits, axis=1)
    if method == "LLLA-EF":
        variance = np.square(phi) @ (1.0 / (h + precision))
    else:
        variance = np.square(phi).sum(axis=1, keepdims=True) / precision
    # Multiclass probit-style approximation, not exact softmax integration.
    return softmax(logits / np.sqrt(1.0 + np.pi * variance / 8.0), axis=1)


def metrics(y, p):
    p = np.clip(p, 1e-15, 1.0)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    # Ten equal-width confidence bins; confidence 1 goes into the last bin.
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(correct[bins == b].mean()
              - confidence[bins == b].mean()) for b in range(10) if np.any(bins == b))
    return {"accuracy": float(correct.mean()),
            "nll": float(-np.log(p[np.arange(len(y)), y]).mean()),
            "brier": float(np.square(p - np.eye(p.shape[1])[y]).sum(axis=1).mean()),
            "ece": float(ece), "confidence": float(confidence.mean())}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0
    y = digits.target
    data_hash = hashlib.sha256(digits.data.tobytes() + y.tobytes()).hexdigest()
    rows, runs, predictions, splits, tuning_rows = [], [], {}, {}, []
    started = time.perf_counter()
    for task in ("ten_class", "open_set"):
        indices = np.arange(len(y)) if task == "ten_class" else np.flatnonzero(y < 5)
        for seed in SEEDS:
            train_val, test = train_test_split(indices, test_size=0.2,
                                              stratify=y[indices], random_state=seed)
            train, val = train_test_split(train_val, test_size=0.25,
                                          stratify=y[train_val], random_state=seed)
            assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
            assert set(train) | set(val) | set(test) == set(indices)
            prefix = f"{task}_{seed}"
            splits[prefix] = {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}
            model = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="lbfgs",
                                  alpha=1.0, max_iter=3000, max_fun=50000,
                                  tol=1e-7, random_state=seed)
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x[train], y[train])
            fit_seconds = time.perf_counter() - t0
            phi_train, phi_val = features(model, x[train]), features(model, x[val])
            weights = last_weights(model)
            np.testing.assert_allclose(predict(phi_val, weights, "MAP"),
                                       model.predict_proba(x[val]), atol=1e-12)
            h = ef_diagonal(phi_train, predict(phi_train, weights, "MAP"), y[train])
            precision_eb = evidence_precision(h, weights)
            # Prespecified grid and ID validation labels only: never test/OOD tuning.
            grid = np.logspace(-2, 6, 81)
            val_nll = [metrics(y[val], predict(phi_val, weights, "ICLA-Val", v))["nll"] for v in grid]
            precision_val = float(grid[np.argmin(val_nll)])
            for v, loss in zip(grid, val_nll):
                tuning_rows.append({"task": task, "seed": seed, "precision": v, "validation_nll": loss})
            np.savez_compressed(output / f"model_{prefix}.npz", w1=model.coefs_[0],
                                b1=model.intercepts_[0], w2=model.coefs_[1],
                                b2=model.intercepts_[1], ef_diagonal=h,
                                precision_eb=precision_eb, precision_val=precision_val)
            run_info = {"task": task, "seed": seed, "n_train": len(train), "n_validation": len(val),
                        "n_test": len(test), "iterations": model.n_iter_, "loss": model.loss_,
                        "train_seconds": fit_seconds, "precision_eb": precision_eb,
                        "precision_val": precision_val, "warnings": [str(w.message) for w in caught]}
            runs.append(run_info)
            sets = {"clean": x[test]}
            if task == "ten_class":
                noise = np.random.default_rng(10000 + seed).normal(0, 0.25, x[test].shape)
                sets["noise025"] = np.clip(x[test] + noise, 0, 1)
            else:
                ood = np.flatnonzero(y >= 5)
                splits[prefix]["ood"] = ood.tolist()
            for condition, test_x in sets.items():
                phi_test = features(model, test_x)
                predictions[f"{prefix}_{condition}_labels"] = y[test]
                for method in METHODS:
                    precision = precision_val if method == "ICLA-Val" else precision_eb
                    p = predict(phi_test, weights, method, precision, h)
                    assert np.isfinite(p).all()
                    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
                    if method.startswith("ICLA"):
                        # The common positive per-image scale preserves logit ordering.
                        assert np.array_equal(p.argmax(axis=1), (phi_test @ weights).argmax(axis=1))
                    result = {"task": task, "seed": seed, "condition": condition, "method": method,
                              **metrics(y[test], p), "ood_auroc": ""}
                    predictions[f"{prefix}_{condition}_{method}"] = p
                    if task == "open_set":
                        p_ood = predict(features(model, x[ood]), weights, method, precision, h)
                        labels = np.r_[np.zeros(len(test)), np.ones(len(ood))]
                        score = np.r_[1 - p.max(axis=1), 1 - p_ood.max(axis=1)]
                        result["ood_auroc"] = float(roc_auc_score(labels, score))
                        predictions[f"{prefix}_ood_{method}"] = p_ood
                    rows.append(result)
            print(json.dumps(run_info), flush=True)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "validation_grid.csv", tuning_rows)
    np.savez_compressed(output / "predictions.npz", **predictions)
    (output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    summary = []
    for task, condition in (("ten_class", "clean"), ("ten_class", "noise025"), ("open_set", "clean")):
        for method in METHODS:
            group = [r for r in rows if r["task"] == task and r["condition"] == condition and r["method"] == method]
            row = {"task": task, "condition": condition, "method": method}
            for metric in ("accuracy", "nll", "brier", "ece", "confidence", "ood_auroc"):
                vals = [r[metric] for r in group if r[metric] != ""]
                row[metric + "_mean"] = float(np.mean(vals)) if vals else ""
                row[metric + "_sd"] = float(np.std(vals, ddof=1)) if vals else ""
            summary.append(row)
    write_csv(output / "summary.csv", summary)
    metadata = {"python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "seeds": SEEDS, "data_sha256": data_hash, "total_samples": len(y),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "seconds": time.perf_counter() - started, "runs": runs,
                "protocol": "Stratified 60/20/20 per seed; fixed pixel /16; one BLAS thread; CPU; no test tuning"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Finished. All results saved to", output.resolve(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
