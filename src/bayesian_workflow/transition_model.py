from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .states import WorkflowState

STATES = [str(s) for s in WorkflowState]

@dataclass
class HierarchicalDirichletTransitionModel:
    """Hierarchical Dirichlet transition model with explicit hyperpriors."""
    hyper_eta: float = 1.0
    concentration_shape: float = 4.0
    concentration_rate: float = 0.5
    alpha_floor: float = 1e-6

    def fit(self, observations: pd.DataFrame) -> "HierarchicalDirichletTransitionModel":
        required = {"stage", "fault_type", "topology", "state_before", "state_after"}
        missing = required - set(observations.columns)
        if missing:
            raise ValueError(f"transition observations missing {sorted(missing)}")
        self.hyper_alpha_: dict[str, np.ndarray] = {}
        self.cell_counts_: dict[tuple[str, str, str, str], np.ndarray] = {}
        for before in STATES:
            subset = observations.loc[observations.state_before == before, "state_after"]
            counts = subset.value_counts().reindex(STATES, fill_value=0).to_numpy(float)
            self.hyper_alpha_[before] = counts + self.hyper_eta
        for key, group in observations.groupby(["stage", "fault_type", "topology", "state_before"]):
            counts = group.state_after.value_counts().reindex(STATES, fill_value=0).to_numpy(float)
            self.cell_counts_[tuple(map(str, key))] = counts
        return self

    def _key(self, stage: str, fault: str, topology: str, before: str) -> tuple[str, str, str, str]:
        return (str(stage), str(fault), str(topology), str(before))

    def alpha(self, stage: str, fault: str, topology: str, before: str) -> np.ndarray:
        mean = self.hyper_alpha_[before] / self.hyper_alpha_[before].sum()
        concentration = self.concentration_shape / self.concentration_rate
        counts = self.cell_counts_.get(self._key(stage, fault, topology, before), np.zeros(len(STATES)))
        return np.maximum(counts + concentration * mean, self.alpha_floor)

    def mean(self, stage: str, fault: str, topology: str, before: str) -> np.ndarray:
        alpha = self.alpha(stage, fault, topology, before)
        return alpha / alpha.sum()

    def draw(self, stage: str, fault: str, topology: str, before: str, rng: np.random.Generator) -> np.ndarray:
        global_mean = rng.dirichlet(self.hyper_alpha_[before])
        concentration = rng.gamma(self.concentration_shape, 1.0 / self.concentration_rate)
        counts = self.cell_counts_.get(self._key(stage, fault, topology, before), np.zeros(len(STATES)))
        return rng.dirichlet(np.maximum(counts + concentration * global_mean, self.alpha_floor))

    def draw_many(self, stage: str, fault: str, topology: str, before: str, draws: int, rng: np.random.Generator) -> np.ndarray:
        return np.vstack([self.draw(stage, fault, topology, before, rng) for _ in range(draws)])

    def summary(self, draws: int = 4_000, seed: int = 20260923) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for (stage, fault, topology, before), counts in sorted(self.cell_counts_.items()):
            samples = self.draw_many(stage, fault, topology, before, draws, rng)
            for j, after in enumerate(STATES):
                rows.append({"stage": stage, "fault_type": fault, "topology": topology,
                    "state_before": before, "state_after": after, "posterior_mean": samples[:, j].mean(),
                    "ci_05": np.quantile(samples[:, j], 0.05), "ci_95": np.quantile(samples[:, j], 0.95),
                    "cell_n": float(counts.sum())})
        return pd.DataFrame(rows)
