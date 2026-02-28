from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from src.tasks.instance.db_bench.task import DBBenchDatasetItem, DBBenchSkillUtility


@dataclass(frozen=True)
class BudgetPrimitive:
    name: str
    token_budget: int
    max_round: int
    tool_budget: int
    stop_enabled: bool

    @property
    def static_cost_proxy(self) -> float:
        # Benchmark-agnostic proxy used for "cheapest-safe" ranking.
        return float(self.token_budget * self.max_round)


# General budget primitives (not benchmark-specific names).
GENERAL_BUDGET_PRIMITIVES: tuple[BudgetPrimitive, ...] = (
    BudgetPrimitive(
        name="budget_64_r1",
        token_budget=64,
        max_round=1,
        tool_budget=0,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_128_r1",
        token_budget=128,
        max_round=1,
        tool_budget=1,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_256_r2",
        token_budget=256,
        max_round=2,
        tool_budget=2,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_512_r3",
        token_budget=512,
        max_round=3,
        tool_budget=4,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_1024_r4",
        token_budget=1024,
        max_round=4,
        tool_budget=4,
        stop_enabled=True,
    ),
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


class TwoHeadLinUCB:
    """
    Two-head linear contextual controller.

    Head 1 predicts correctness probability (in [0, 1], clipped).
    Head 2 predicts normalized cost (>= 0, clipped), typically scaled by budget_target.

    Decision rule per action a:
        score(a) = p_correct(a) - lambda * cost(a) + alpha * uncertainty(a)
    """

    def __init__(self, n_actions: int, feature_dim: int, alpha: float):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.a_inv_list: list[list[list[float]]] = []
        self.b_correct_list: list[list[float]] = []
        self.b_cost_list: list[list[float]] = []
        for _ in range(n_actions):
            a_inv = [[0.0 for _ in range(feature_dim)] for _ in range(feature_dim)]
            for i in range(feature_dim):
                a_inv[i][i] = 1.0
            self.a_inv_list.append(a_inv)
            self.b_correct_list.append([0.0 for _ in range(feature_dim)])
            self.b_cost_list.append([0.0 for _ in range(feature_dim)])

    @staticmethod
    def _sigmoid(value: float) -> float:
        # Stable enough for the current scale of linear outputs.
        if value >= 0:
            exp_neg = math.exp(-value)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(value)
        return exp_pos / (1.0 + exp_pos)

    def predict_all(self, x: list[float]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for action_idx in range(self.n_actions):
            a_inv = self.a_inv_list[action_idx]
            theta_correct = _mat_vec(a_inv, self.b_correct_list[action_idx])
            theta_cost = _mat_vec(a_inv, self.b_cost_list[action_idx])
            raw_correct = _dot(theta_correct, x)
            predicted_correct = self._sigmoid(raw_correct)
            predicted_cost = max(_dot(theta_cost, x), 0.0)
            a_inv_x = _mat_vec(a_inv, x)
            uncertainty = math.sqrt(max(_dot(x, a_inv_x), 0.0))
            out.append(
                {
                    "raw_correct": raw_correct,
                    "predicted_correct": predicted_correct,
                    "predicted_cost": predicted_cost,
                    "uncertainty": uncertainty,
                }
            )
        return out

    def select_action(
        self,
        x: list[float],
        lambda_value: float,
        candidate_action_indices: Sequence[int] | None = None,
    ) -> tuple[int, dict[str, float]]:
        best_action = 0
        best_score = float("-inf")
        best_meta: dict[str, float] = {}
        predictions = self.predict_all(x)
        if candidate_action_indices is None:
            candidate_action_indices = tuple(range(self.n_actions))
        for action_idx in candidate_action_indices:
            pred = predictions[action_idx]
            predicted_correct = pred["predicted_correct"]
            predicted_cost = pred["predicted_cost"]
            uncertainty = pred["uncertainty"]
            utility = predicted_correct - (lambda_value * predicted_cost)
            score = utility + (self.alpha * uncertainty)
            if score > best_score:
                best_action = action_idx
                best_score = score
                best_meta = {
                    "score": score,
                    "utility": utility,
                    "raw_correct": pred["raw_correct"],
                    "predicted_correct": predicted_correct,
                    "predicted_cost": predicted_cost,
                    "uncertainty": uncertainty,
                }
        return best_action, best_meta

    def select_cheapest_safe_action(
        self,
        x: list[float],
        correctness_threshold: float,
        action_cost_hint: Sequence[float],
        lambda_value: float = 0.0,
        candidate_action_indices: Sequence[int] | None = None,
    ) -> tuple[int, dict[str, float]]:
        """
        Choose an action from the safe set (predicted correctness >= threshold) by
        maximizing utility = p_correct - lambda * cost_hint.
        If none are safe, fallback to the same utility rule over all candidates.
        """
        predictions = self.predict_all(x)
        if candidate_action_indices is None:
            candidate_action_indices = tuple(range(self.n_actions))

        def utility(idx: int) -> float:
            pred = predictions[idx]
            return float(pred["predicted_correct"]) - (
                float(lambda_value) * float(action_cost_hint[idx])
            )

        safe_indices = []
        for idx in candidate_action_indices:
            if predictions[idx]["predicted_correct"] >= correctness_threshold:
                safe_indices.append(idx)
        if safe_indices:
            best_action = max(
                safe_indices,
                key=utility,
            )
            pred = predictions[best_action]
            return best_action, {
                "policy": "cheapest_safe",
                "threshold": correctness_threshold,
                "lambda_value": float(lambda_value),
                "raw_correct": pred["raw_correct"],
                "predicted_correct": pred["predicted_correct"],
                "predicted_cost": pred["predicted_cost"],
                "cost_hint": float(action_cost_hint[best_action]),
                "uncertainty": pred["uncertainty"],
                "utility": utility(best_action),
                "score": utility(best_action),
            }
        fallback_idx = max(candidate_action_indices, key=utility)
        pred = predictions[fallback_idx]
        return fallback_idx, {
            "policy": "fallback_utility",
            "threshold": correctness_threshold,
            "lambda_value": float(lambda_value),
            "raw_correct": pred["raw_correct"],
            "predicted_correct": pred["predicted_correct"],
            "predicted_cost": pred["predicted_cost"],
            "cost_hint": float(action_cost_hint[fallback_idx]),
            "uncertainty": pred["uncertainty"],
            "utility": utility(fallback_idx),
            "score": utility(fallback_idx),
        }

    def update(
        self,
        action_idx: int,
        x: list[float],
        correct_label: float,
        cost_label: float,
    ) -> None:
        a_inv = self.a_inv_list[action_idx]
        b_correct = self.b_correct_list[action_idx]
        b_cost = self.b_cost_list[action_idx]
        a_inv_x = _mat_vec(a_inv, x)
        denom = 1.0 + _dot(x, a_inv_x)
        # Sherman-Morrison update for A_inv where A <- A + x x^T
        for i in range(self.feature_dim):
            for j in range(self.feature_dim):
                a_inv[i][j] -= (a_inv_x[i] * a_inv_x[j]) / denom
        for i in range(self.feature_dim):
            b_correct[i] += correct_label * x[i]
            b_cost[i] += cost_label * x[i]


class DBBenchFeatureAdapter:
    def __init__(self):
        self.skill_list = DBBenchSkillUtility.get_all_skill_list()

    @property
    def feature_dim(self) -> int:
        # Dense features:
        # [bias, instr, rows, cols, skill_cnt, instr*skill_cnt, rows*skill_cnt, cols*skill_cnt,
        #  row_bucket_50, row_bucket_200, instr_bucket_300]
        # + one-hot skills
        return 11 + len(self.skill_list)

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
            norm_instruction * norm_skill_count,
            norm_rows * norm_skill_count,
            norm_cols * norm_skill_count,
            1.0 if row_count > 50 else 0.0,
            1.0 if row_count > 200 else 0.0,
            1.0 if instruction_len > 300 else 0.0,
        ]
        for skill in self.skill_list:
            features.append(1.0 if skill in sample_skills else 0.0)
        return features
