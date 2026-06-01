"""Quantum state-discrimination classifier for generator attribution.

Each generator ``g`` is summarized by one or more reference density matrices --
the mean (or k-means prototypes) of its training images' patch-ensemble density
matrices, each itself a valid density matrix. A query image's density matrix
``rho_x`` is attributed to the generator it is *least distinguishable* from,
under a quantum distinguishability measure:

  - ``"fidelity"``: argmax_g Uhlmann fidelity F(rho_x, rho_g)
  - ``"trace"``:    argmin_g trace distance T(rho_x, rho_g)
  - ``"hs"``:       argmax_g Born-rule overlap Tr(rho_g rho_x) -- the mean
                    measurement probability of the query's patches under the
                    reference state (equals the Hilbert-Schmidt inner product
                    for our real symmetric matrices).
  - ``"pgm"``:      the pretty-good (square-root) measurement -- a genuine POVM
                    E_g = rho_bar^{-1/2} (p_g rho_g) rho_bar^{-1/2}, with
                    rho_bar = sum_g p_g rho_g; predict argmax_g Tr(E_g rho_x).

This casts attribution as a quantum hypothesis-testing problem rather than a
generic feature classifier. The pretty-good measurement is near-optimal for
multi-state discrimination (and optimal for two equiprobable states), tying the
classifier to the Helstrom/Holevo theory of distinguishing quantum states.

Because every generator's images share a large "natural-texture" bulk, the
class-mean references are nearly identical and the discriminative signal lives
in a small difference subspace (see docs/results). Two knobs counter this
*without* leaving the quantum framing:

  - ``pgm_rcond`` regularizes the PGM whitening with a *relative* spectral floor
    (a pseudo-inverse square root). With the old absolute clip, rho_bar^{-1/2}
    amplified near-empty null directions by ~1e15; the relative floor instead
    whitens only inside the populated subspace, where the difference signal it
    is meant to surface actually lives.
  - ``n_prototypes`` replaces each single mean reference with k-means prototypes
    in density-matrix (Hilbert-Schmidt) space, so a multi-modal generator is not
    collapsed to one diluted average.

Inputs are flattened ``(d * d,)`` density matrices so the estimator drops into
the same cross-validation harness as the numpy feature classifiers.
"""

from __future__ import annotations

import numpy as np

from qif_attribution.features.patch_spectrum import state_fidelity, trace_distance

RULES = ("fidelity", "trace", "hs", "pgm")


class QuantumStateDiscriminator:
    """Attribute density matrices to the least-distinguishable reference state."""

    def __init__(
        self,
        dim: int,
        rule: str = "fidelity",
        *,
        n_prototypes: int = 1,
        pgm_rcond: float = 1e-3,
        eps: float = 1e-10,
        seed: int = 0,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r} (choose from {', '.join(RULES)})")
        if n_prototypes < 1:
            raise ValueError("n_prototypes must be >= 1")
        self.dim = dim
        self.rule = rule
        self.n_prototypes = n_prototypes
        self.pgm_rcond = pgm_rcond
        self.eps = eps
        self.seed = seed

    def fit(self, features: np.ndarray, labels: list[str]) -> QuantumStateDiscriminator:
        array = np.asarray(features, dtype=np.float64)
        if array.shape[0] != len(labels):
            raise ValueError("number of feature rows must match number of labels")
        matrices = array.reshape(array.shape[0], self.dim, self.dim)
        label_array = np.asarray(labels)
        n_total = matrices.shape[0]
        rng = np.random.default_rng(self.seed)
        self.classes_ = sorted(set(labels))
        self.references_: dict[str, np.ndarray] = {}
        self.priors_: dict[str, float] = {}
        # prototypes_[cls] = [(weight, matrix), ...]; weights over ALL prototypes
        # of ALL classes sum to 1, so sum_j weight_j == prior of the class.
        self.prototypes_: dict[str, list[tuple[float, np.ndarray]]] = {}
        for cls in self.classes_:
            members = matrices[label_array == cls]
            self.references_[cls] = members.mean(axis=0)
            self.priors_[cls] = members.shape[0] / n_total
            self.prototypes_[cls] = self._prototypes(members, n_total, rng)
        if self.rule == "pgm":
            self._build_pgm()
        return self

    def _prototypes(
        self, members: np.ndarray, n_total: int, rng: np.random.Generator
    ) -> list[tuple[float, np.ndarray]]:
        """Per-class prototypes as (weight, matrix); a class mean when k == 1."""

        k = min(self.n_prototypes, members.shape[0])
        if k <= 1:
            return [(members.shape[0] / n_total, members.mean(axis=0))]
        centers, counts = _kmeans(members.reshape(members.shape[0], -1), k, rng)
        return [
            (count / n_total, center.reshape(self.dim, self.dim))
            for center, count in zip(centers, counts, strict=True)
            if count > 0
        ]

    def _inv_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Pseudo-inverse square root with a relative spectral floor.

        Eigen-directions below ``pgm_rcond * max_eigenvalue`` are dropped instead
        of inverted, so whitening stays inside the populated subspace.
        """

        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        values = np.clip(values, 0.0, None)
        floor = self.pgm_rcond * values.max()
        inv = np.where(values > floor, 1.0 / np.sqrt(np.maximum(values, self.eps)), 0.0)
        return (vectors * inv) @ vectors.T

    def _build_pgm(self) -> None:
        rho_bar = sum(
            weight * matrix
            for protos in self.prototypes_.values()
            for weight, matrix in protos
        )
        inv_sqrt = self._inv_sqrt(rho_bar)
        # E_g = sum_j rho_bar^{-1/2} (w_j rho_{g,j}) rho_bar^{-1/2}. For full-rank
        # rho_bar these sum to the identity; under the relative floor they sum to
        # the projector onto the populated subspace, which leaves argmax intact.
        self.povm_ = {
            cls: sum(inv_sqrt @ (weight * matrix) @ inv_sqrt for weight, matrix in protos)
            for cls, protos in self.prototypes_.items()
        }

    def predict(self, features: np.ndarray) -> list[str]:
        if not hasattr(self, "classes_"):
            raise RuntimeError("model must be fit before predict")
        array = np.asarray(features, dtype=np.float64)
        matrices = array.reshape(array.shape[0], self.dim, self.dim)
        return [self._predict_one(rho) for rho in matrices]

    def _predict_one(self, rho: np.ndarray) -> str:
        if self.rule == "pgm":
            return max(self.classes_, key=lambda c: float(np.sum(self.povm_[c] * rho)))
        if self.rule == "fidelity":
            return max(
                self.classes_,
                key=lambda c: max(state_fidelity(rho, m) for _, m in self.prototypes_[c]),
            )
        if self.rule == "trace":
            return min(
                self.classes_,
                key=lambda c: min(trace_distance(rho, m) for _, m in self.prototypes_[c]),
            )
        # hs: Born-rule overlap Tr(rho_g rho_x); score a class by its best prototype.
        return max(
            self.classes_,
            key=lambda c: max(float(np.sum(m * rho)) for _, m in self.prototypes_[c]),
        )


def _kmeans(
    data: np.ndarray, k: int, rng: np.random.Generator, *, max_iter: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm in flattened-matrix space. Returns (centers, counts)."""

    n = data.shape[0]
    centers = data[rng.choice(n, size=k, replace=False)].copy()
    assign = None
    sq_norms = (data**2).sum(1)
    for _ in range(max_iter):
        # ||a - b||^2 = |a|^2 + |b|^2 - 2 a.b; the |a|^2 term is constant in argmin.
        dist = sq_norms[:, None] + (centers**2).sum(1)[None, :] - 2.0 * data @ centers.T
        new_assign = dist.argmin(1)
        if assign is not None and np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for j in range(k):
            sel = assign == j
            if sel.any():
                centers[j] = data[sel].mean(0)
    counts = np.bincount(assign, minlength=k)
    return centers, counts
