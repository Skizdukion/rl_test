import numpy as np
import torch
from typing import Union

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update_model_params(src: torch.nn.Module, dest: torch.nn.Module, tau=1e-3):
    """Soft updates model parameters (θ_dest = τ * θ_src + (1 - τ) * θ_src)."""
    for dest_param, src_param in zip(dest.parameters(), src.parameters()):
        dest_param.data.copy_(tau * src_param.data + (1.0 - tau) * dest_param.data)


def cart_pole_adjust_reward(next_state, reward) -> Union[np.float64, np.array]:
    angle = next_state[2] if len(next_state.shape) == 1 else next_state[:, 2]
    position = next_state[0] if len(next_state.shape) == 1 else next_state[:, 0]
    return reward - np.abs(angle) / 0.418 - np.abs(position) / 4.8


# Function to compute cumulative discounted rewards
def compute_cumulative_rewards(rewards, gamma):
    discounts = np.power(gamma, np.arange(len(rewards)))
    return np.array(
        [
            np.sum(rewards[t:] * discounts[: len(rewards) - t])
            for t in range(len(rewards))
        ]
    )


def epsilon_gen(eps_start=1.0, eps_decay=0.99999, eps_min=0.05):
    """Generator function for Ɛ and its decay (e.g., exploration via Ɛ-greedy policy)."""
    eps = eps_start
    while True:
        yield eps
        eps = max(eps * eps_decay, eps_min)
