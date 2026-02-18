from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from src.tasks.instance.db_bench.task import DBBenchDatasetItem, DBBenchSkillUtility


@dataclass(frozen=True)
class ComputeProfile:
    name: str
    max_new_tokens: int
    max_round: int


DBBENCH_COMPUTE_PROFILES: tuple[ComputeProfile, ...] = (
    ComputeProfile(name="low", max_new_tokens=128, max_round=1),
    ComputeProfile(name="mid", max_new_tokens=256, max_round=2),
    ComputeProfile(name="high", max_new_tokens=512, max_round=3),
)


def _dot(v1: Sequence[float], v2: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def _mat_vec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> list[float]:
    return [_dot(row, vec) for row in mat]


class LinUCB:
    def __init__(self, n_actions: int, feature_dim: int, alpha: float):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.a_inv_list: list[list[list[float]]] = []
        self.b_list: list[list[float]] = []
        for _ in range(n_actions):
            a_inv = [[0.0 for _ in range(feature_dim)] for _ in range(feature_dim)]
            for i in range(feature_dim):
                a_inv[i][i] = 1.0
            self.a_inv_list.append(a_inv)
            self.b_list.append([0.0 for _ in range(feature_dim)])

    def select_action(self, x: list[float]) -> tuple[int, float]:
        best_action = 0
        best_score = float("-inf")
        for action_idx in range(self.n_actions):
            a_inv = self.a_inv_list[action_idx]
            b = self.b_list[action_idx]
            theta = _mat_vec(a_inv, b)
            exploit = _dot(theta, x)
            a_inv_x = _mat_vec(a_inv, x)
            explore = self.alpha * math.sqrt(max(_dot(x, a_inv_x), 0.0))
            score = exploit + explore
            if score > best_score:
                best_action = action_idx
                best_score = score
        return best_action, best_score

    def update(self, action_idx: int, x: list[float], reward: float) -> None:
        a_inv = self.a_inv_list[action_idx]
        b = self.b_list[action_idx]
        a_inv_x = _mat_vec(a_inv, x)
        denom = 1.0 + _dot(x, a_inv_x)
        # Sherman-Morrison update for A_inv where A <- A + x x^T
        for i in range(self.feature_dim):
            for j in range(self.feature_dim):
                a_inv[i][j] -= (a_inv_x[i] * a_inv_x[j]) / denom
        for i in range(self.feature_dim):
            b[i] += reward * x[i]


class DBBenchFeatureAdapter:
    def __init__(self):
        self.skill_list = DBBenchSkillUtility.get_all_skill_list()

    @property
    def feature_dim(self) -> int:
        # [bias, log_instruction_len, log_row_count, log_col_count, skill_count] + one-hot skills
        return 5 + len(self.skill_list)

    def build(self, item: DBBenchDatasetItem) -> list[float]:
        instruction_len = len(item.instruction)
        row_count = len(item.table_info.row_list)
        col_count = len(item.table_info.column_info_list)
        sample_skills = set(item.skill_list)
        skill_den = max(len(self.skill_list), 1)

        # Keep dense features in roughly [0, 1] to stabilize LinUCB.
        # log1p(22026) ~= 10, so divide by 10 for a soft normalization.
        norm_instruction = min(math.log1p(instruction_len) / 10.0, 1.0)
        norm_rows = min(math.log1p(row_count) / 10.0, 1.0)
        norm_cols = min(math.log1p(col_count) / 10.0, 1.0)
        norm_skill_count = min(float(len(sample_skills)) / float(skill_den), 1.0)

        features: list[float] = [
            1.0,
            norm_instruction,
            norm_rows,
            norm_cols,
            norm_skill_count,
        ]
        for skill in self.skill_list:
            features.append(1.0 if skill in sample_skills else 0.0)
        return features
