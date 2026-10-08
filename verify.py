"""Numerical and saved-result checks independent of the training loop."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from experiment import ef_diagonal, evidence_precision, METHODS


def main():
    root = Path(__file__).resolve().parent / "results"
    rng = np.random.default_rng(1729)
    phi, weights = rng.normal(size=(7, 4)), rng.normal(size=(4, 3))
    y = np.arange(7) % 3
    gradients = np.zeros((7, 4, 3))
    eps = 1e-5
    for j in range(4):
        for k in range(3):
            perturb = np.zeros_like(weights)
            perturb[j, k] = eps
            plus = -np.log(softmax(phi @ (weights + perturb), axis=1)[np.arange(7), y])
            minus = -np.log(softmax(phi @ (weights - perturb), axis=1)[np.arange(7), y])
            gradients[:, j, k] = (plus - minus) / (2 * eps)
    h = ef_diagonal(phi, softmax(phi @ weights, axis=1), y)
    np.testing.assert_allclose(h, np.square(gradients).sum(axis=0), rtol=1e-6, atol=1e-8)
    lam = evidence_precision(h, weights)
    derivative = 0.5 * (lam * np.square(weights).sum() - (h / (h + lam)).sum())
    assert abs(derivative) < 1e-5, derivative
    # Check analytic last-layer logit variance with actual Gaussian weight samples.
    draws = rng.normal(size=(100000, 4, 3)) / np.sqrt(lam)
    samples = np.einsum("d,sdc->sc", phi[0], draws)
    np.testing.assert_allclose(samples.var(axis=0), np.square(phi[0]).sum() / lam, rtol=0.015)
    data = np.load(root / "predictions.npz")
    rows = list(csv.DictReader((root / "metrics.csv").open(encoding="utf-8")))
    assert len(rows) == 60
    for row in rows:
        key = f"{row['task']}_{row['seed']}_{row['condition']}"
        p, labels = data[key + "_" + row['method']], data[key + "_labels"]
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
        assert abs(accuracy_score(labels, p.argmax(axis=1)) - float(row['accuracy'])) < 1e-12
        assert abs(log_loss(labels, p) - float(row['nll'])) < 1e-12
        brier = np.mean(np.sum((p - np.eye(p.shape[1])[labels]) ** 2, axis=1))
        assert abs(brier - float(row['brier'])) < 1e-12
        ece = 0.0
        conf = p.max(axis=1)
        for b in range(10):
            mask = (conf >= b / 10) & ((conf < (b + 1) / 10) if b < 9 else (conf <= 1))
            if mask.any():
                ece += abs(np.sum((p.argmax(axis=1)[mask] == labels[mask]) - conf[mask])) / len(labels)
        assert abs(ece - float(row['ece'])) < 1e-12
        if row['task'] == 'open_set':
            po = data[f"open_set_{row['seed']}_ood_{row['method']}"]
            auc = roc_auc_score(np.r_[np.zeros(len(p)), np.ones(len(po))],
                                np.r_[1-p.max(axis=1), 1-po.max(axis=1)])
            assert abs(auc - float(row['ood_auroc'])) < 1e-12
    splits = json.loads((root / 'splits.json').read_text())
    for key, split in splits.items():
        a, b, c = (set(split[s]) for s in ('train', 'validation', 'test'))
        assert not (a & b or a & c or b & c)
        if key.startswith('open_set'):
            assert not ((a | b | c) & set(split['ood']))
    for row in csv.DictReader((root / 'summary.csv').open(encoding='utf-8')):
        group = [r for r in rows if all(r[k] == row[k] for k in ['task', 'condition', 'method'])]
        assert len(group) == 5
        for metric in ['accuracy','nll','brier','ece','confidence','ood_auroc']:
            vals = [float(r[metric]) for r in group if r[metric]]
            if vals:
                assert abs(np.mean(vals) - float(row[metric+'_mean'])) < 1e-12
                assert abs(np.std(vals, ddof=1) - float(row[metric+'_sd'])) < 1e-12
    metadata = json.loads((root / 'metadata.json').read_text())
    assert not any(r['warnings'] for r in metadata['runs'])
    code_hash = hashlib.sha256((root.parent/'experiment.py').read_bytes()).hexdigest()
    assert code_hash == metadata['code_sha256']
    result = {'status': 'PASS', 'checked_metric_rows': len(rows), 'checked_summary_rows':12,
              'checks': ['finite-difference empirical Fisher', 'evidence stationary point',
                         'Gaussian logit variance Monte Carlo sanity check', 'probability normalization',
                         'independent accuracy/NLL/Brier/ECE/AUROC recalculation',
                         'split and OOD separation', 'summary mean and sample SD',
                         'no convergence warnings', 'experiment code SHA-256']}
    (root/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
