from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .states import WorkflowState


STATES = [str(s) for s in WorkflowState]


@dataclass
class HierarchicalDirichletTransitionModel:
    """Partially pooled Bayesian multistate transition estimator.

    Each stage/fault/control cell receives a Dirichlet prior centered on the
    corresponding state-specific global transition distribution. This
    conjugate specification is transparent and reproducible for sparse cells.
    """

    prior_strength: float = 8.0
    alpha_floor: float = 0.25

    def fit(self, observations: pd.DataFrame) -> "HierarchicalDirichletTransitionModel":
        required = {"stage", "fault_type", "topology", "state_before", "state_after"}
        missing = required - set(observations.columns)
        if missing:
            raise ValueError(f"transition observations missing {sorted(missing)}")
        self.global_alpha_: dict[str, np.ndarray] = {}
        for before in STATES:
            subset = observations.loc[observations["state_before"] == before, "state_after"]
            counts = subset.value_counts().reindex(STATES, fill_value=0).to_numpy(float)
            probs = (counts + 1.0) / (counts.sum() + len(STATES))
            self.global_alpha_[before] = np.maximum(probs * self.prior_strength, self.alpha_floor)
        self.cell_alpha_: dict[tuple[str, str, str, str], np.ndarray] = {}
        grouped = observations.groupby(["stage", "fault_type", "topology", "state_before"])
        for key, group in grouped:
            counts = group["state_after"].value_counts().reindex(STATES, fill_value=0).to_numpy(float)
            self.cell_alpha_[tuple(map(str, key))] = self.global_alpha_[str(key[3])] + counts
        return self

    def alpha(self, stage: str, fault: str, topology: str, before: str) -> np.ndarray:
        return self.cell_alpha_.get((stage, fault, topology, before), self.global_alpha_[before])

    def mean(self, stage: str, fault: str, topology: str, before: str) -> np.ndarray:
        alpha = self.alpha(stage, fault, topology, before)
        return alpha / alpha.sum()

    def draw(self, stage: str, fault: str, topology: str, before: str, rng: np.random.Generator) -> np.ndarray:
        return rng.dirichlet(self.alpha(stage, fault, topology, before))

    def summary(self, draws: int = 4_000, seed: int = 20260923) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows: list[dict[str, object]] = []
        for (stage, fault, topology, before), alpha in sorted(self.cell_alpha_.items()):
            samples = rng.dirichlet(alpha, size=draws)
            for j, after in enumerate(STATES):
                rows.append({
                    "stage": stage, "fault_type": fault, "topology": topology,
                    "state_before": before, "state_after": after,
                    "posterior_mean": samples[:, j].mean(),
                    "ci_05": np.quantile(samples[:, j], 0.05),
                    "ci_95": np.quantile(samples[:, j], 0.95),
                    "cell_n": float(alpha.sum() - self.global_alpha_[before].sum()),
                })
        return pd.DataFrame(rows)
