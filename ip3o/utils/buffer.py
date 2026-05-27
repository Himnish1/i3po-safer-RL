from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        out[i] = running
    return out


@dataclass
class RolloutBuffer:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    reward_values: List[float] = field(default_factory=list)
    cost_values: List[float] = field(default_factory=list)

    def add(self, obs, action, reward, cost, done, log_prob, reward_value, cost_value) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.costs.append(float(cost))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.reward_values.append(float(reward_value))
        self.cost_values.append(float(cost_value))

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.costs.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.reward_values.clear()
        self.cost_values.clear()

    def _gae(self, signal, values):
        signal = np.asarray(signal, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        deltas = signal + self.gamma * values[1:] * (1.0 - dones) - values[:-1]
        adv = np.zeros_like(signal, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(signal))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values[:-1]
        return adv, returns

    def _discounted_returns(self, signal, bootstrap_value: float = 0.0):
        signal = np.asarray(signal, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        returns = np.zeros_like(signal, dtype=np.float32)
        running = float(bootstrap_value)
        for t in reversed(range(len(signal))):
            running = signal[t] + self.gamma * (1.0 - dones[t]) * running
            returns[t] = running
        return returns

    def get(self, device: torch.device, last_vr: float = 0.0, last_vc: float = 0.0) -> Dict[str, torch.Tensor]:
        # Compute reward GAE (advantage) and returns
        reward_adv, reward_ret = self._gae(self.rewards, self.reward_values + [last_vr])
        # Compute cost GAE and returns (use GAE for consistency with reward)
        cost_adv, cost_ret = self._gae(self.costs, self.cost_values + [last_vc])
        # Normalize reward advantages only; preserve cost_adv magnitude for L_C computation
        reward_adv = (reward_adv - reward_adv.mean()) / (reward_adv.std() + 1e-8)

        batch = {
            "obs": torch.as_tensor(np.asarray(self.obs), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.asarray(self.actions), dtype=torch.float32, device=device),
            "log_probs": torch.as_tensor(np.asarray(self.log_probs), dtype=torch.float32, device=device),
            "reward_adv": torch.as_tensor(reward_adv, dtype=torch.float32, device=device),
            "cost_adv": torch.as_tensor(cost_adv, dtype=torch.float32, device=device),
            "reward_returns": torch.as_tensor(reward_ret, dtype=torch.float32, device=device),
            "cost_returns": torch.as_tensor(cost_ret, dtype=torch.float32, device=device),
        }
        return batch
