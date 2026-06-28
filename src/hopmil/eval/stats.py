"""Bayesian comparison of two aggregators from their cross-validated scores.

This is the inferential layer of the study (see ``docs/METHODOLOGY.md``). We do
NOT use frequentist significance tests: in k-fold CV the per-fold scores are
correlated (overlapping training sets), which inflates type-I error for the
paired t-test (Dietterich 1998), and a non-significant result cannot *assert*
equivalence — which is exactly the claim "attention can be replaced by Hopfield"
needs.

Instead we use the **correlated Bayesian t-test** (Corani & Benavoli 2015) with
the **Nadeau & Bengio (2003)** variance correction for the train-set overlap,
plus a **ROPE** (Region of Practical Equivalence). Integrating the posterior of
the mean difference over the three regions yields three probabilities summing
to 1: P(left), P(rope), P(right).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BayesResult:
    """Posterior mass in each region of the mean difference (in AUC points).

    Three probabilities summing to 1:

    ``p_a_better`` = P(A practically better than B).
    ``p_rope``     = P(A and B practically equivalent, |diff| <= rope).
    ``p_b_better`` = P(B practically better than A).
    """

    name_a: str
    name_b: str
    rope: float
    p_a_better: float
    p_rope: float
    p_b_better: float

    def verdict(self, threshold: float = 0.95) -> str:
        if self.p_rope >= threshold:
            return f"{self.name_a} ~= {self.name_b} (equivalentes)"
        if self.p_a_better >= threshold:
            return f"{self.name_a} > {self.name_b}"
        if self.p_b_better >= threshold:
            return f"{self.name_b} > {self.name_a}"
        return "inconclusivo"

    def as_row(self) -> dict:
        return {
            "a": self.name_a,
            "b": self.name_b,
            "rope": self.rope,
            "p_a_better": self.p_a_better,
            "p_rope": self.p_rope,
            "p_b_better": self.p_b_better,
            "verdict": self.verdict(),
        }


def compare_bayesian(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    name_a: str,
    name_b: str,
    rope: float,
    runs: int,
) -> BayesResult:
    """Run the correlated Bayesian t-test on paired per-fold scores.

    Args:
        scores_a, scores_b: per-fold scores of the two aggregators, **paired**
            (same fold at the same index). Length = runs * folds.
        rope: half-width of the ROPE, in the same units as the scores (AUC).
        runs: number of CV *repeats* — baycomp uses it with the fold count it
            infers to apply the Nadeau & Bengio correlation correction.
    """
    from baycomp import two_on_single

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"score arrays must be paired/same-length: {a.shape} vs {b.shape}")

    # baycomp convention: two_on_single(x, y) -> (P(x better), P(rope), P(y better)).
    p_a_better, p_rope, p_b_better = two_on_single(a, b, rope=rope, runs=runs)
    return BayesResult(
        name_a=name_a,
        name_b=name_b,
        rope=rope,
        p_a_better=float(p_a_better),
        p_rope=float(p_rope),
        p_b_better=float(p_b_better),
    )
