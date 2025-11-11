from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.callbacks.callback import Callback, CallbackArguments
from src.agents.controller_agent import ControllerAgent
from src.typings import Session


class ControllerTrainingCallback(Callback):
    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        lambda_cost: float = 0.001,
        device: str = "mps",
    ):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lambda_cost = lambda_cost
        self.device = torch.device(device)
        self.optimizer: Optional[torch.optim.Optimizer] = None

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _maybe_init_opt(self, agent: ControllerAgent) -> None:
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=self.lr)

    def _success_reward(self, session: Session) -> float:
        evaluation_record = getattr(session, "evaluation_record", None)
        if evaluation_record is None:
            return 0.0
        outcome = getattr(evaluation_record, "outcome", None)
        if isinstance(outcome, (int, float)):
            return float(outcome)
        if isinstance(outcome, str):
            if outcome.lower() in {"success", "correct", "pass"}:
                return 1.0
            return 0.0
        score = getattr(evaluation_record, "score", None)
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        agent = callback_args.session_context.agent
        if not isinstance(agent, ControllerAgent):
            return

        traj: List[Dict[str, Any]] = agent.episode_buffer
        if not traj:
            return

        self._maybe_init_opt(agent)
        assert self.optimizer is not None

        session = callback_args.current_session
        success = self._success_reward(session)

        rewards: List[float] = []
        for i, step in enumerate(traj):
            r = -self.lambda_cost * float(step["cost"])
            if i == len(traj) - 1:
                r += success
            rewards.append(r)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        returns = torch.zeros_like(rewards_t)
        g = 0.0
        for t in range(len(rewards_t) - 1, -1, -1):
            g = rewards_t[t] + self.gamma * g
            returns[t] = g

        if returns.numel() > 1:
            std = returns.std(unbiased=False)
            if std > 0:
                returns = (returns - returns.mean()) / (std + 1e-8)

        states_np = np.stack([step["state"] for step in traj], axis=0)
        states = torch.from_numpy(states_np).to(self.device, dtype=torch.float32)
        actions = torch.tensor(
            [step["action"] for step in traj],
            dtype=torch.long,
            device=self.device,
        )

        values, log_probs, entropy = agent.policy.evaluate_actions(states, actions)
        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = (returns - values).pow(2).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        agent.episode_buffer = []
