from abc import ABC, abstractmethod
import numpy as np
import torch
from trajectory_segment import LearningBatch
from utils import DEVICE
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class IAction(ABC):
    @abstractmethod
    def act(self, state: np.array, action_mask=None, eps=0.0) -> int:
        pass


class RandomAgent(IAction):
    def __init__(self):
        pass

    def act(self, state: np.array, action_mask=None, eps=0.0):
        if action_mask is not None:
            # Filter valid actions using the mask
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) == 0:
                raise ValueError("No valid actions available.")
            return np.random.choice(valid_actions)
        else:
            # If no mask is provided, choose a random action from the full range
            return np.random.randint(0, len(state))


class AgentPPO(nn.Module, IAction):
    def __init__(self, state_size, action_size, lr: float = 1e-4, weight_mul=1e-3):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.eps = 0
        self.critic = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(DEVICE)

    def get_action_probs(
        self, states, action_mask=None
    ) -> torch.distributions.Distribution:
        states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)

        is_add_batch_dim = False
        if len(states_tensor.shape) == 1:
            is_add_batch_dim = True
            states_tensor = states_tensor.unsqueeze(0)  # Add batch dimension

        action_logits = self.actor(
            states_tensor
        )  # Get the logits from the actor network

        # Apply action mask if provided
        if action_mask is not None:
            # Set the logits of the masked actions to a very large negative number
            action_mask = torch.tensor(action_mask).to(DEVICE)
            if is_add_batch_dim:
                action_mask = action_mask.unsqueeze(0)

            action_logits = action_logits + ((1 - action_mask) * -1e10)

        # Create the Categorical distribution
        action_probs = torch.distributions.Categorical(logits=action_logits)

        return action_probs

    def sample_action(self, states, action_mask=None):
        probs = self.get_action_probs(states, action_mask)
        action = probs.sample()
        return action, probs.log_prob(action)

    def set_eps(self, eps):
        self.eps = eps

    @torch.no_grad()
    def act(self, state: np.array, action_mask=None):

        action, _ = self.sample_action(
            torch.from_numpy(state).unsqueeze(0).to(DEVICE), action_mask
        )
        return action.cpu().numpy()[0]

    def get_value(self, states):
        states_tensor = (
            torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )
        return self.critic(states_tensor)

    def eval_action(self, states, action):
        probs = self.get_action_probs(states)
        return probs.log_prob(action), probs.entropy()

    def learn(
        self,
        batch: LearningBatch,
        entropy_coeff=0.01,
        vf_coeff=0.5,
        clip_coeff=0.1,
        max_grad_norm=0.75,
    ):
        newlogprobs, entropy = self.eval_action(batch.states, batch.actions)

        logratio = newlogprobs - batch.logprobs
        ratio = logratio.exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff)
        advantages = batch.advantages

        L_entropy = entropy_coeff * entropy.mean()
        L_clipped = -torch.min(advantages * ratio, advantages * clipped_ratio).mean()
        L_actor = L_clipped - L_entropy

        newvalues = self.get_value(batch.states).view(-1)
        L_critic = F.mse_loss(newvalues, batch.returns)

        L_ppo = L_actor + L_critic * vf_coeff
        self.optimizer.zero_grad()
        L_ppo.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.optimizer.step()

    def save(self, file_path: str):
        """
        Save the model's state_dict to the given file path.
        """
        torch.save(self.state_dict(), file_path)
        # print(f"Model saved to {file_path}")

    def load(self, file_path: str):
        """
        Load the model's state_dict from the given file path.
        """
        state_dict = torch.load(file_path)
        self.load_state_dict(state_dict)
        # print(f"Model loaded from {file_path}")

    def clone(self):
        state_dict = self.state_dict()
        new = AgentPPO(self.state_size, self.action_size)
        new.load_state_dict(state_dict)
        new.train(False)
        return new


# class AgentDQN:
#     def __init__(
#         self,
#         state_size: int,
#         action_size: int,
#         gamma: float = 0.99,
#         tau: float = 1e-3,
#         lr: float = 1e-4,
#         batch_size: int = 32,
#         learn_every: int = 4,
#         update_target_every: int = 2,
#         preload_file: str = None,
#     ):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.gamma = gamma
#         self.tau = tau
#         self.lr = lr
#         self.batch_size = batch_size
#         self.learn_every = learn_every
#         self.update_target_every = update_target_every
#         self.t_learn_step = 0
#         self.t_update_target_step = 0

#         self.memory = ReplayBuffer()
#         self.qnetwork_local = QResnet(state_size, action_size).to(DEVICE)
#         self.qnetwork_target = QResnet(state_size, action_size).to(DEVICE)
#         self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
#         self.qnetwork_target.eval()
#         self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=self.lr)

#         if preload_file is not None:
#             print(f"Loading pre-trained model: {preload_file}")
#             self.qnetwork_local.load_state_dict(
#                 torch.load(preload_file, map_location=DEVICE)
#             )

#     def step(self, state, action, reward, next_state, done):
#         """Tells the agent to make a step: record experience and possibly learn."""
#         self.memory.add(state, action, reward, next_state, done)
#         self.t_learn_step = (self.t_learn_step + 1) % self.learn_every
#         if self.t_learn_step == 0:
#             if len(self.memory) > self.batch_size:
#                 self.learn()
#         self.t_update_target_step = (
#             self.t_update_target_step + 1
#         ) % self.update_target_every
#         if self.t_update_target_step == 0:
#             soft_update_model_params(
#                 self.qnetwork_local, self.qnetwork_target, self.tau
#             )

#     @torch.no_grad
#     def act(self, state: np.array, action_mask=None, eps=0.0):
#         """Makes the agent take an action for the state passed as input."""
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         action_values = self.qnetwork_local(state_tensor)

#         if action_mask is not None:
#             action_mask = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
#             action_values[action_mask == 0] = -float("Inf")

#         if random.random() > eps:
#             return np.argmax(action_values.cpu().data.numpy())
#         else:
#             valid_actions = torch.nonzero(action_mask.squeeze(0), as_tuple=True)[
#                 0
#             ]  # Find valid actions
#             return random.choice(valid_actions.cpu().data.numpy())

#     def learn(self, experiences):
#         """Executes one learning step for the agent."""
#         states, actions, rewards, next_states, dones = experiences

#         with torch.no_grad():
#             target_action_values = self.qnetwork_target(
#                 next_states
#             ).detach()  # (batch_s, action_s)
#             max_action_values = target_action_values.amax(
#                 1, keepdim=True
#             )  # (batch_size, 1)
#             Q_targets = rewards + (
#                 self.gamma * max_action_values * (1 - dones)
#             )  # (batch_size, 1)

#         predictions = self.qnetwork_local(states)
#         actions = actions.long()
#         Q_expected = predictions.gather(1, actions)
#         loss = F.huber_loss(Q_targets, Q_expected)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#     def add_memory(self, states, actions, rewards, next_states, dones, env: Game):
#         lengths = [len(arr) for arr in [states, actions, rewards, next_states, dones]]

#         if len(set(lengths)) != 1:
#             # return False
#             raise ValueError("All arrays must have the same length")

#         for i in range(len(states)):
#             self.memory.add(
#                 env.get_encoded_state(states[i]),
#                 actions[i],
#                 rewards[i],
#                 env.get_encoded_state(next_states[i]),
#                 dones[i],
#             )

#     def multi_learn(self, learn_size=5):
#         for _ in range(learn_size):
#             experiences = self.memory.sample(1024)
#             self.learn(experiences)

#     def save(self, filepath):
#         torch.save(
#             {
#                 "qnetwork": self.qnetwork_local.state_dict(),
#                 "qnetwork_optimizer": self.optimizer.state_dict(),
#             },
#             filepath,
#         )
#         print(f"Models saved to {filepath}")

#     def load(self, filepath):
#         checkpoint = torch.load(filepath)
#         self.qnetwork_local.load_state_dict(checkpoint["qnetwork"])
#         self.optimizer.load_state_dict(checkpoint["qnetwork_optimizer"])
#         self.qnetwork_target.load_state_dict(checkpoint["qnetwork"])
#         print(f"Models loaded from {filepath}")

#     def clone(self):
#         """Creates a clone of the current AgentDQN instance."""
#         clone_agent = AgentDQN(
#             state_size=self.state_size,
#             action_size=self.action_size,
#             gamma=self.gamma,
#             tau=self.tau,
#             lr=self.lr,
#             batch_size=self.batch_size,
#             learn_every=self.learn_every,
#             update_target_every=self.update_target_every,
#         )

#         # Clone the Q-networks' state dicts
#         clone_agent.qnetwork_local.load_state_dict(self.qnetwork_local.state_dict())
#         clone_agent.qnetwork_target.load_state_dict(self.qnetwork_target.state_dict())

#         # Set the cloned networks to evaluation mode
#         clone_agent.qnetwork_local.eval()
#         clone_agent.qnetwork_target.eval()

#         # Copy other relevant attributes
#         clone_agent.memory = (
#             self.memory
#         )  # We can choose to clone memory or just share the same one
#         clone_agent.t_learn_step = self.t_learn_step
#         clone_agent.t_update_target_step = self.t_update_target_step

#         # Set the optimizer parameters (optional)
#         clone_agent.optimizer = optim.RMSprop(
#             clone_agent.qnetwork_local.parameters(), lr=self.lr
#         )

#         return clone_agent


# class AgentDDPG:
#     def __init__(
#         self,
#         state_size,
#         action_size,
#         start_mem_size=128,
#         gamma=0.99,
#         lr_actor=1e-4,
#         lr_critic=1e-3,
#         exploration_noise_scale=0.1,
#     ):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.start_mem_size = start_mem_size
#         self.gamma = gamma
#         self.exploration_noise_scale = exploration_noise_scale

#         self.actor = PolicyResnet(state_size, action_size).to(DEVICE)
#         self.actor_target = PolicyResnet(state_size, action_size).to(DEVICE)
#         self.actor_target.load_state_dict(self.actor.state_dict())
#         self.actor_target.eval()
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

#         self.critic = QResnet(state_size, action_size).to(DEVICE)
#         self.critic_target = QResnet(state_size, action_size).to(DEVICE)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         self.critic_target.eval()
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

#         self.step_count = 0
#         self.learn_step = 5
#         self.memory = ReplayBuffer()

#     @torch.no_grad
#     def act(self, state: np.array, action_mask=None):
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         action_probs = self.actor(state_tensor)
#         # If an action mask is provided, apply it
#         if action_mask is not None:
#             action_mask = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
#             action_probs[action_mask == 0] = -float("Inf")

#         action = torch.multinomial(torch.exp(action_probs), 1).item()
#         return action

#     def add_memory(self, states, actions, rewards, next_states, dones, env: Game):
#         lengths = [len(arr) for arr in [states, actions, rewards, next_states, dones]]

#         if len(set(lengths)) != 1:
#             # return False
#             raise ValueError("All arrays must have the same length")

#         for i in range(len(states)):
#             self.memory.add(
#                 env.get_encoded_state(states[i]),
#                 actions[i],
#                 rewards[i],
#                 env.get_encoded_state(next_states[i]),
#                 dones[i],
#             )

#     def multi_learn(self, learn_size=5):
#         for _ in range(learn_size):
#             experiences = self.memory.sample()
#             self.learn(experiences)

#     def step(self, state, action, reward, next_state, done):
#         self.memory.add(state, action, reward, next_state, done)
#         self.step_count += 1
#         if self.step_count % self.learn_step == 0:
#             if len(self.memory) > self.start_mem_size:
#                 experiences = self.memory.sample()
#                 self.learn(experiences)

#     def learn(self, experiences):
#         states, actions, rewards, next_states, dones = experiences

#         with torch.no_grad():
#             # Get next action from target actor (for next states)
#             actions_next = self.actor_target(next_states)

#             # Get Q-value from the critic target using next states and next actions
#             Q_targets_next = self.critic_target(next_states, actions_next)

#             # Compute target Q-value
#             Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

#         # Ensure actions are of the correct type (long, for indexing)
#         actions = actions.long()

#         # Create a zero tensor for one-hot encoding
#         actions_one_hot = torch.zeros(states.size(0), self.action_size).to(DEVICE)

#         # Scatter 1s into the actions positions for each batch
#         actions_one_hot.scatter_(1, actions, 1)
#         # Get Q-values from critic network for current states and actions
#         Q_values = self.critic(states, actions_one_hot)

#         # Compute critic loss (Mean Squared Error between predicted and target Q-values)
#         critic_loss = F.mse_loss(Q_values, Q_targets)

#         # Zero gradients for the critic optimizer, perform backward pass, and step
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         # Update actor network (maximize the Q-value predicted by the critic for the actions taken)
#         actor_loss = -self.critic(states, self.actor(states)).mean()

#         # Zero gradients for the actor optimizer, perform backward pass, and step
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # Soft update the target networks for both actor and critic
#         soft_update_model_params(self.critic, self.critic_target)
#         soft_update_model_params(self.actor, self.actor_target)

#     def clone(self):
#         """
#         Creates a clone of the current AgentDDPG instance.
#         The clone will be set to evaluation mode and will not require training.
#         """
#         clone_agent = AgentDDPG(
#             state_size=self.state_size,
#             action_size=self.action_size,
#             start_mem_size=self.start_mem_size,
#             gamma=self.gamma,
#             lr_actor=self.actor_optimizer.param_groups[0]["lr"],
#             lr_critic=self.critic_optimizer.param_groups[0]["lr"],
#             exploration_noise_scale=self.exploration_noise_scale,
#         )

#         # Copy actor and critic networks' weights
#         clone_agent.actor.load_state_dict(self.actor.state_dict())
#         clone_agent.actor_target.load_state_dict(self.actor_target.state_dict())
#         clone_agent.critic.load_state_dict(self.critic.state_dict())
#         clone_agent.critic_target.load_state_dict(self.critic_target.state_dict())

#         # Set all networks to evaluation mode
#         clone_agent.actor.eval()
#         clone_agent.actor_target.eval()
#         clone_agent.critic.eval()
#         clone_agent.critic_target.eval()

#         # Disable gradient calculations
#         for param in clone_agent.actor.parameters():
#             param.requires_grad = False
#         for param in clone_agent.actor_target.parameters():
#             param.requires_grad = False
#         for param in clone_agent.critic.parameters():
#             param.requires_grad = False
#         for param in clone_agent.critic_target.parameters():
#             param.requires_grad = False

#         clone_agent.step_count = self.step_count
#         clone_agent.learn_step = self.learn_step

#         return clone_agent

#     def save(self, filepath):
#         torch.save(
#             {
#                 "actor": self.actor.state_dict(),
#                 "actor_optimizer": self.actor_optimizer.state_dict(),
#                 "critic": self.critic.state_dict(),  # Save the state_dict of the critic
#                 "critic_optimizer": self.critic_optimizer.state_dict(),
#             },
#             filepath,
#         )
#         print(f"Models saved to {filepath}")

#     def load(self, filepath):
#         checkpoint = torch.load(filepath)
#         self.actor.load_state_dict(checkpoint["actor"])
#         self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
#         self.critic.load_state_dict(checkpoint["critic"])
#         self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
#         print(f"Models loaded from {filepath}")

#     # @staticmethod
#     # def load(filepath, state_size, action_size):
#     #     """
#     #     Static method to load an AgentDDPG model from a checkpoint.
#     #     Returns a new instance of AgentDDPG with the model parameters loaded.
#     #     """
#     #     # Load checkpoint
#     #     checkpoint = torch.load(filepath)

#     #     # Create a new AgentDDPG instance
#     #     agent = AgentDDPG(
#     #         state_size,
#     #         action_size,
#     #     )

#     #     # Load the saved model parameters into the new instance
#     #     agent.actor.load_state_dict(checkpoint["actor"])
#     #     agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
#     #     agent.critic.load_state_dict(checkpoint["critic"])
#     #     agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

#     #     print(f"Models loaded from {filepath}")
#     #     return agent
