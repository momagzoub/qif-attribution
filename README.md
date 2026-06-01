# Quantum Fingerprint Attribution

Attribute AI-generated images to their source generator. A numpy-only core of
physics features (radial-FFT + patch-ensemble *density matrices*), a frozen-CNN
ceiling, and a quantum **state-discrimination** classifier — all evaluated under
leakage-safe 5-fold group cross-validation (folds split by prompt) on a
1,024-image, 4-generator pilot. Chance = 0.25.

## Results (small set, 5-fold group CV)

| Method | Accuracy |
| --- | --- |
| Physics features (radial-FFT + density), k-NN | ~0.82–0.84 |
| Frozen ResNet-50 / ConvNeXt-Large + logreg (ceiling) | ~0.88–0.90 |
| Quantum state discrimination — single mean reference (trace) | 0.53 |
| Quantum state discrimination — 16 k-means prototypes (trace / fidelity) | **0.84** |

Modeling each generator as a *mixture* of sub-state prototypes — rather than one
diluted mean density matrix — lifts the quantum classifier from 0.53 to 0.84,
matching the physics features and approaching the CNN ceiling (verified
leakage-free: folds are prompt-disjoint). Honest caveat: nearest-prototype
discrimination tends toward k-NN in Hilbert–Schmidt space as the prototype count
grows, and 0.84 is the top of a prototype sweep; the pre-registered single-state
("textbook" Helstrom) number is 0.53.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
```

Analysis benchmarks live in `scripts/analysis/` (e.g.
`quantum_discriminator_small.py`). They need the image set, which is not
committed — data stays out of git.

## Layout

```text
src/qif_attribution/   package — features, models, eval (numpy-only core)
scripts/analysis/      CV benchmarks: physics, deep ceiling, quantum discriminator
tests/                 unit + integration + regression
```
