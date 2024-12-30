from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class TrajectorySegment:
    states: torch.Tensor  # (M, R, S_dim)
    actions: torch.Tensor  # (M, R)
    logprobs: torch.Tensor  # (M, R)
    values: torch.Tensor  # (M, R)
    rewards: torch.Tensor  # (M, R)
    dones: torch.Tensor  # (M, R)

    next_start_state: torch.Tensor  # (M, S_dim)

    def __len__(self):
        return self.states.shape[0]


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


class Batcher:
    def __init__(
        self,
        seg: TrajectorySegment,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        n_mini_batches: int,
    ):
        # self.batch_size = (
        #     seg.states.shape[0] * seg.states.shape[1]
        # )  # rollout_len * num_bots
        self.batch_size = seg.states.shape[0]
        self.mini_batch_size = int(self.batch_size // n_mini_batches)
        # self.experiences = LearningBatch(
        #     *Batcher.flatten(
        #         (seg.states, seg.actions, seg.logprobs, advantages, returns)
        #     )
        # )
        self.experiences = LearningBatch(
            seg.states, seg.actions, seg.logprobs, advantages, returns
        )

    def shuffle(self):
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        return Batcher.MiniBatchIterator(
            self.experiences, indices, self.mini_batch_size
        )

    @staticmethod
    def flatten(t: tuple[torch.Tensor, ...]) -> tuple:
        return tuple(x.flatten(0, 1) for x in t)

    class MiniBatchIterator:
        def __init__(
            self, experiences: LearningBatch, indices: list[int], mini_batch_size: int
        ):
            self.experiences = experiences
            self.indices = indices
            self.mini_batch_size = mini_batch_size
            self.start = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.start >= len(self.experiences):
                raise StopIteration()
            start = self.start
            end = start + self.mini_batch_size
            inds = self.indices[start:end]
            self.start = end
            return self.experiences[inds]
