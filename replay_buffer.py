from dataclasses import dataclass
from typing import NamedTuple
from collections import deque

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Experience(NamedTuple):
    """A single step / experience of an agent stored in the replay buffer."""

    state: np.array
    action: float
    reward: float
    next_state: np.array
    done: bool


class ReplayBuffer:
    """Simple replay buffer for off-policy deep reinforcement learning algorithms.

    IMPORTANT: This ReplayBuffer is specifically tuned for the DDPG / TD3 / SAC algorithms in these
    lectures. In particular, the action space is a single float scalar. If you want to adapt those
    algorithms to different environments, you will need to update this code accordingly as well.
    """

    def __init__(self, buffer_size=int(2e4)):
        """Initializes the buffer with an internal deque of size `buffer_size`."""
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Stores a single step / experience of an agent."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size: int = 128):
        """Randomly selects `batch_size` items from the buffer, stacks them, and returns them."""
        all_indices = np.arange(len(self.memory))
        selection = np.random.choice(all_indices, size=batch_size)
        return self.unpack(selection)

    def unpack(self, selection):
        """Given the `selection` of experiences, returns them as a tuple of stacked values.

        This is convenient for the usage in the various learning algorithms so that they don't have
        to do it themselves.
        """
        experiences = [e for i in selection if (e := self.memory[i]) is not None]
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.stack(states)).float().to(DEVICE)
        actions = (
            torch.from_numpy(np.vstack(actions)).float().to(DEVICE)
        )  # NOTE: float scalar!
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones, dtype=np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


@dataclass
class LearningBatch:
    states: torch.Tensor  # (B, S_dim)
    actions: torch.Tensor  # (B, A_dim)
    logprobs: torch.Tensor  # (B)
    advantages: torch.Tensor  # (B)
    returns: torch.Tensor  # (B)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, key):
        return LearningBatch(
            self.states[key],
            self.actions[key],
            self.logprobs[key],
            self.advantages[key],
            self.returns[key],
        )
