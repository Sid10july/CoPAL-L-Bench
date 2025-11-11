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
    def __init__(
        self,
        base_agent,
        state_dim: int,
        n_actions: int,
        lambda_cost: float = 0.001,
        device: str = "cpu",
    ):
        super().__init__()

        if isinstance(base_agent, dict) and "module" in base_agent:
            base_agent_factory = GeneralInstanceFactory.model_validate(base_agent)
            self.base_agent: Agent = base_agent_factory.create()
        else:
            self.base_agent = base_agent

        self.lambda_cost = lambda_cost
        self.device = torch.device(device)
        self.policy = ControllerPolicy(state_dim, n_actions).to(self.device)
        self.config_list: list[dict] = [{} for _ in range(n_actions)]
        self.episode_buffer: list[dict] = []

    def set_config_list(self, config_list: list[dict]) -> None:
        self.config_list = config_list

    def encode_state(self, session) -> np.ndarray:
        return np.array([float(session.sample_index)], dtype=np.float32)

    def apply_config_to_base_agent(self, config: dict) -> None:
        if hasattr(self.base_agent, "set_config"):
            self.base_agent.set_config(config)

    def inference(self, session) -> None:
        state_vec = self.encode_state(session)
        state = torch.tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        value, action, log_prob = self.policy.act(state)
        idx = int(action.item())

        config = self.config_list[idx] if self.config_list else {}
        self.apply_config_to_base_agent(config)

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

    def _inference(self, chat_history):
        return self.base_agent._inference(chat_history)
