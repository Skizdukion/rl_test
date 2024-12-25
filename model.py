import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import DEVICE


class PolicyResnet(nn.Module):
    def __init__(
        self, state_size, action_size, num_resBlocks=4, num_hidden=64, device=DEVICE
    ):
        super().__init__()

        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.seq = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * state_size, action_size),
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x = self.startBlock(x)

        for resBlock in self.backBone:
            x = resBlock(x)

        return torch.softmax(self.seq(x), dim=-1)


class QResnet(nn.Module):
    def __init__(
        self, state_size, action_size, num_resBlocks=4, num_hidden=64, device=DEVICE
    ):
        super().__init__()

        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.seq = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * state_size, action_size),
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x = self.startBlock(x)

        for resBlock in self.backBone:
            x = resBlock(x)

        return self.seq(x)


# class QResnet(nn.Module):
#     def __init__(
#         self, state_size, action_size, num_resBlocks=4, num_hidden=64, device=DEVICE
#     ):
#         super().__init__()

#         self.device = device

#         # State feature extractor (ConvNet)
#         self.startBlock = nn.Sequential(
#             nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU(),
#         )

#         self.backBone = nn.ModuleList(
#             [ResBlock(num_hidden) for i in range(num_resBlocks)]
#         )

#         # Flatten and reduce state features
#         self.state_fc = nn.Sequential(
#             nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Flatten(),  # Flatten before combining with actions
#         )

#         # Action feature extractor
#         self.action_fc = nn.Linear(action_size, 128)

#         # Final layers to predict Q-values after combining state and action
#         self.q_value_fc = nn.Sequential(
#             nn.Linear(32 * state_size + 128, 256),  # Combine state and action features
#             nn.ReLU(),
#             nn.Linear(256, 1),  # Output single Q-value
#         )

#         self.to(self.device)

#     def forward(self, state: torch.Tensor, action: torch.Tensor):
#         """
#         Args:
#             state: Tensor of shape [batch_size, 3, height, width]
#             action: Tensor of shape [batch_size, action_size]
#         """
#         # Process state through the conv backbone
#         x = self.startBlock(state)
#         for resBlock in self.backBone:
#             x = resBlock(x)
#         state_features = self.state_fc(x)

#         # Process actions through a linear layer
#         action_features = self.action_fc(action)

#         # Concatenate state and action features
#         combined_features = torch.cat([state_features, action_features], dim=1)

#         # Predict Q-value
#         q_value = self.q_value_fc(combined_features)

#         return q_value


# class QResnet(nn.Module):
#     def __init__(self, state_size, action_size, num_resBlocks, num_hidden, device):
#         super().__init__()

#         self.device = device

#         self.startBlock = nn.Sequential(
#             nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU(),
#         )

#         self.backBone = nn.ModuleList(
#             [ResBlock(num_hidden) for i in range(num_resBlocks)]
#         )

#         self.seq = nn.Sequential(
#             nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(32 * state_size, action_size),
#         )

#         self.to(self.device)

#     def forward(self, x: torch.Tensor):
#         x = self.startBlock(x)

#         for resBlock in self.backBone:
#             x = resBlock(x)

#         return self.seq(x)


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()

        self.con1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.con2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor):
        residual = x
        x = F.relu(self.bn1(self.con1(x)))
        x = self.bn2(self.con2(x))
        x += residual
        x = F.relu(x)
        return x
