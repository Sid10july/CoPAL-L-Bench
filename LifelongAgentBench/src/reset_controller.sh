#!/usr/bin/env bash
set -euo pipefail

# Run this from the src/ directory (where run_experiment.py lives)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Root: $ROOT"

echo "Removing old controller-related files (if they exist)..."
rm -f "$ROOT/agents/controller_agent.py"
rm -f "$ROOT/callbacks/controller_training_callback.py"
rm -rf "$ROOT/controller"

echo "Recreating agents/controller_agent.py..."
cat > "$ROOT/agents/controller_agent.py" <<'PY'
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.agents import Agent
from src.typings import GeneralInstanceFactory


class ControllerPolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def act(self, state: torch.Tensor):
        x = self.net(state)
        logits = self.action_head(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_head(x).squeeze(-1)
        return value, action, log_prob

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        x = self.net(states)
        logits = self.action_head(x)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_head(x).squeeze(-1)
        return values, log_probs, entropy


class ControllerAgent(Agent):
    """
    Wraps a base agent and learns a policy over 'configs' that control how
    expensive the base agent is (CoT depth, tools, etc).

    The 'base_agent' parameter can be either:
      - a dict with 'module' and 'parameters' (GeneralInstanceFactory style), or
      - an already-instantiated Agent.
    """

    def __init__(
        self,
        base_agent,
        state_dim: int,
        n_actions: int,
        lambda_cost: float = 0.001,
        device: str = "cpu",
    ):
        # If base_agent is a config dict, instantiate it via GeneralInstanceFactory
        if isinstance(base_agent, dict) and "module" in base_agent:
            base_agent_factory = GeneralInstanceFactory.model_validate(base_agent)
            self.base_agent: Agent = base_agent_factory.create()
        else:
            self.base_agent = base_agent

        self.lambda_cost = lambda_cost
        self.device = torch.device(device)
        self.policy = ControllerPolicy(state_dim, n_actions).to(self.device)

        # Discrete configs; you can later fill these with real modes (cheap/expensive, etc.)
        self.config_list: list[dict] = [{} for _ in range(n_actions)]
        self.episode_buffer: list[dict] = []

    def set_config_list(self, config_list: list[dict]) -> None:
        self.config_list = config_list

    def encode_state(self, session) -> np.ndarray:
        """
        Build a numeric state vector for the controller.
        For now: just use the sample index as a scalar feature.
        """
        return np.array([float(session.sample_index)], dtype=np.float32)

    def apply_config_to_base_agent(self, config: dict) -> None:
        """
        Apply a chosen config to the base agent (e.g., set CoT depth, tools, etc.)
        This assumes base_agent (or its subclass) implements set_config().
        """
        if hasattr(self.base_agent, "set_config"):
            self.base_agent.set_config(config)

    def inference(self, session) -> None:
        """
        One controller step:
          - encode state
          - sample an action/config
          - run the base agent under that config
          - log cost and store (state, action, cost) for training
        """
        state_vec = self.encode_state(session)
        state = torch.tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        value, action, log_prob = self.policy.act(state)
        idx = int(action.item())

        config = self.config_list[idx] if self.config_list else {}
        self.apply_config_to_base_agent(config)

        # Measure cost as delta in tokens (or whatever you track in the session)
        prev_tokens = int(getattr(session, "total_tokens", 0))
        self.base_agent.inference(session)
        new_tokens = int(getattr(session, "total_tokens", prev_tokens))
        step_cost = max(new_tokens - prev_tokens, 0)

        self.episode_buffer.append(
            {
                "state": state_vec.astype(np.float32),
                "action": idx,
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
                "cost": float(step_cost),
            }
        )
PY

echo "Recreating callbacks/controller_training_callback.py..."
cat > "$ROOT/callbacks/controller_training_callback.py" <<'PY'
from typing import Any, Dict, List, Optional
import torch

from src.callbacks.callback import Callback, CallbackArguments
from src.agents.controller_agent import ControllerAgent
from src.typings import Session


class ControllerTrainingCallback(Callback):
    """
    Trains the ControllerAgent's policy after each completed sample
    using a simple REINFORCE-style update with a cost penalty.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        lambda_cost: float = 0.001,
        device: str = "cpu",
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
        # Only one instance of this callback should exist per run
        return True

    def _maybe_init_opt(self, agent: ControllerAgent) -> None:
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=self.lr)

    def _success_reward(self, session: Session) -> float:
        """
        Extract a scalar success reward from the session's evaluation_record.
        You can customize this to match your metrics.
        """
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
            # If the current agent is not a ControllerAgent, do nothing
            return

        traj: List[Dict[str, Any]] = agent.episode_buffer
        if not traj:
            return

        self._maybe_init_opt(agent)
        assert self.optimizer is not None

        session = callback_args.current_session
        success = self._success_reward(session)

        # Reward per step: -lambda_cost * cost, plus final success on the last step.
        rewards: List[float] = []
        for i, step in enumerate(traj):
            r = -self.lambda_cost * float(step["cost"])
            if i == len(traj) - 1:
                r += success
            rewards.append(r)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Compute discounted returns
        returns = torch.zeros_like(rewards_t)
        g = 0.0
        for t in range(len(rewards_t) - 1, -1, -1):
            g = rewards_t[t] + self.gamma * g
            returns[t] = g

        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = torch.tensor(
            [step["state"] for step in traj],
            dtype=torch.float32,
            device=self.device,
        )
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

        # Clear trajectory for next sample
        agent.episode_buffer = []
PY

echo "Done. New controller files written."
