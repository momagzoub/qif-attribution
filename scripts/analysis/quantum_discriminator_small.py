#!/usr/bin/env python
"""Quantum state-discrimination attribution on the small set.

The novel quantum angle: instead of feeding density-matrix *summary statistics*
to a generic classifier, we attribute images by *quantum state discrimination*.
Each image becomes a patch-ensemble density matrix; each generator gets one (or
a few k-means) reference density matrices (the train-fold mean of its images'
matrices); a held-out image is assigned to the generator it is least
distinguishable from, under four measures -- Uhlmann fidelity, trace distance,
the Born-rule / Hilbert-Schmidt overlap Tr(rho_g rho_x), and the pretty-good
(square-root) measurement (a genuine POVM, near-optimal for multi-state
discrimination). Everything runs under the same 5-fold group CV (by prompt_id)
as the other benchmarks, so the numbers are directly comparable to the physics
features (~0.82-0.84) and the deep ceiling (~0.88-0.90).

Two knobs counter the mean-reference dilution that the class means suffer (every
generator's images share a big "natural-texture" bulk, so the means are nearly
identical): a *relative* spectral floor ``pgm_rcond`` on the PGM whitening, and
``n_prototypes`` k-means references per class. The headline grid fixes both at
their pre-registered defaults; a clearly-labelled exploratory sweep then probes
their sensitivity on the strongest config. Swept maxima are exploratory (they
pay a multiple-comparisons tax) and are NOT to be read as the headline number.

Run from the repo root (needs the analysis extra for pillow):

    PYTHONPATH=src python scripts/analysis/quantum_discriminator_small.py

Per-image density matrices are cached as data/processed/small/qrho_<config>.npy
so reruns are fast. Writes data/processed/small/metrics_quantum_disc_small.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qif_attribution.data.manifest import ManifestRecord, load_manifest
from qif_attribution.eval.crossval import cross_validate
from qif_attribution.features.extract import load_image_array
from qif_attribution.features.patch_spectrum import density_matrix_from_image
from qif_attribution.models.quantum_discriminator import RULES, QuantumStateDiscriminator

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "interim" / "scale" / "small" / "manifest.csv"
SMALL = ROOT / "data" / "processed" / "small"

N_SPLITS = 5
SEED = 1729

# Each density-matrix config: (patch_size, color, highpass). dim = (3 if color
# else 1) * patch_size**2 -- the side length of the per-image density matrix.
CONFIGS = {
    "gray_p8": {"patch_size": 8, "color": False, "highpass": False},
    "gray_hp_p8": {"patch_size": 8, "color": False, "highpass": True},
    "rgb_p8": {"patch_size": 8, "color": True, "highpass": False},
}

# Pre-registered defaults for the headline grid.
DEFAULT_RCOND = 1e-3
DEFAULT_PROTOTYPES = 1

# Exploratory sweep (single strongest config; clearly labelled, not the headline).
SWEEP_CONFIG = "gray_hp_p8"
RCOND_GRID = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4)
PROTO_GRID = (2, 4, 8, 16)


def config_dim(cfg: dict) -> int:
    return (3 if cfg["color"] else 1) * cfg["patch_size"] ** 2


def build_rho_stack(records: list[ManifestRecord], cfg: dict) -> np.ndarray:
    """Flattened per-image density matrices, one row per manifest image."""

    rows = [
        density_matrix_from_image(
            load_image_array(ROOT / record.relative_path),
            patch_size=cfg["patch_size"],
            color=cfg["color"],
            highpass=cfg["highpass"],
        ).reshape(-1)
        for record in records
    ]
    return np.vstack(rows)


def load_or_build(records: list[ManifestRecord], name: str, cfg: dict) -> np.ndarray:
    dim = config_dim(cfg)
    cache = SMALL / f"qrho_{name}.npy"
    if cache.exists():
        stack = np.load(cache)
        if stack.shape == (len(records), dim * dim):
            return stack
    stack = build_rho_stack(records, cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, stack)
    return stack


def evaluate(
    stack: np.ndarray, labels: list[str], groups: list[str], dim: int, **kwargs
) -> tuple[dict, dict]:
    """One 5-fold group-CV run; returns (accuracy summary, macro-F1 summary)."""

    result = cross_validate(
        stack,
        labels,
        groups,
        lambda: QuantumStateDiscriminator(dim=dim, **kwargs),
        n_splits=N_SPLITS,
        seed=SEED,
    )
    return result.summary("accuracy"), result.summary("macro_f1")


def _acc_cells(acc: dict) -> tuple[str, str]:
    return (
        f"[{acc['ci_low']:.3f},{acc['ci_high']:.3f}]",
        f"{acc['min']:.3f}-{acc['max']:.3f}",
    )


def run_headline(
    stacks: dict[str, np.ndarray], labels: list[str], groups: list[str]
) -> list[tuple]:
    rows: list[tuple[str, int, str, dict, dict]] = []
    for name, cfg in CONFIGS.items():
        dim = config_dim(cfg)
        for rule in RULES:
            acc, f1 = evaluate(
                stacks[name],
                labels,
                groups,
                dim,
                rule=rule,
                n_prototypes=DEFAULT_PROTOTYPES,
                pgm_rcond=DEFAULT_RCOND,
            )
            rows.append((name, dim, rule, acc, f1))
    rows.sort(key=lambda row: row[3]["mean"], reverse=True)

    header = (
        f"{'config':<12}{'dim':>5}  {'rule':<10}"
        f"{'acc':>7}{'95% CI':>17}{'macroF1':>9}{'folds(min-max)':>18}"
    )
    print("HEADLINE GRID (pre-registered defaults: "
          f"n_prototypes={DEFAULT_PROTOTYPES}, pgm_rcond={DEFAULT_RCOND:g})")
    print(header)
    print("-" * len(header))
    for name, dim, rule, acc, f1 in rows:
        ci, span = _acc_cells(acc)
        print(
            f"{name:<12}{dim:>5}  {rule:<10}{acc['mean']:>7.3f}{ci:>17}"
            f"{f1['mean']:>9.3f}{span:>18}"
        )
    best = rows[0]
    print(
        f"\nBEST headline cell (selected from grid -- exploratory): {best[0]} "
        f"({best[1]}-dim) + {best[2]} -> acc {best[3]['mean']:.3f} "
        f"95% CI [{best[3]['ci_low']:.3f}, {best[3]['ci_high']:.3f}]"
    )
    return rows


def run_sweep(
    stacks: dict[str, np.ndarray], labels: list[str], groups: list[str]
) -> dict:
    dim = config_dim(CONFIGS[SWEEP_CONFIG])
    stack = stacks[SWEEP_CONFIG]
    print(f"\nEXPLORATORY SWEEP on {SWEEP_CONFIG} ({dim}-dim) -- multiple comparisons; "
          "not the headline.")

    print("\n  (a) PGM whitening floor pgm_rcond (rule=pgm, n_prototypes=1)")
    print(f"  {'rcond':>8}{'acc':>8}{'95% CI':>18}{'macroF1':>9}")
    rcond_rows = []
    for rcond in RCOND_GRID:
        acc, f1 = evaluate(stack, labels, groups, dim, rule="pgm", pgm_rcond=rcond)
        ci, _ = _acc_cells(acc)
        print(f"  {rcond:>8g}{acc['mean']:>8.3f}{ci:>18}{f1['mean']:>9.3f}")
        rcond_rows.append({"rcond": rcond, "accuracy_mean": acc["mean"],
                           "accuracy_ci": [acc["ci_low"], acc["ci_high"]],
                           "macro_f1_mean": f1["mean"]})

    print("\n  (b) k-means prototypes per class n_prototypes (pgm_rcond default)")
    print(f"  {'rule':<10}{'k':>3}{'acc':>8}{'95% CI':>18}{'macroF1':>9}")
    proto_rows = []
    for rule in RULES:
        for k in PROTO_GRID:
            acc, f1 = evaluate(
                stack, labels, groups, dim, rule=rule, n_prototypes=k, pgm_rcond=DEFAULT_RCOND
            )
            ci, _ = _acc_cells(acc)
            print(f"  {rule:<10}{k:>3}{acc['mean']:>8.3f}{ci:>18}{f1['mean']:>9.3f}")
            proto_rows.append({"rule": rule, "n_prototypes": k, "accuracy_mean": acc["mean"],
                               "accuracy_ci": [acc["ci_low"], acc["ci_high"]],
                               "macro_f1_mean": f1["mean"]})

    return {"config": SWEEP_CONFIG, "dim": dim, "rcond": rcond_rows, "prototypes": proto_rows}


def main() -> int:
    records = load_manifest(MANIFEST)
    labels = [record.generator_id for record in records]
    groups = [record.prompt_id for record in records]
    n_classes = len(set(labels))
    print(
        f"images={len(records)} classes={n_classes} groups(prompt_id)={len(set(groups))} "
        f"folds={N_SPLITS} chance={1 / n_classes:.3f}\n"
    )

    stacks = {name: load_or_build(records, name, cfg) for name, cfg in CONFIGS.items()}

    headline = run_headline(stacks, labels, groups)
    sweep = run_sweep(stacks, labels, groups)

    payload = {
        "n_splits": N_SPLITS,
        "seed": SEED,
        "chance": 1 / n_classes,
        "defaults": {"n_prototypes": DEFAULT_PROTOTYPES, "pgm_rcond": DEFAULT_RCOND},
        "headline_grid": [
            {
                "config": name,
                "dim": dim,
                "rule": rule,
                "accuracy_mean": acc["mean"],
                "accuracy_ci": [acc["ci_low"], acc["ci_high"]],
                "macro_f1_mean": f1["mean"],
            }
            for name, dim, rule, acc, f1 in headline
        ],
        "exploratory_sweep": sweep,
    }
    out_path = SMALL / "metrics_quantum_disc_small.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote quantum-discriminator metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
