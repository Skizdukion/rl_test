from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor

import random
import numpy as np
import torch
from agent import AgentPPO, IAction, RandomAgent
from env import TicTacToe
from replay_buffer import DEVICE
from trajectory_segment import Batcher, TrajectorySegment

# from multiprocessing import Pool
import time


@torch.no_grad()
def basic_advantages_and_returns(
    segment: TrajectorySegment, next_return: torch.Tensor, gamma=0.9
):
    returns = torch.zeros_like(segment.rewards).to(DEVICE).detach()

    roll_out_size = segment.rewards.shape[1]

    for t in reversed(range(roll_out_size)):
        is_terminal = 1 - segment.dones[:, t]
        returns[:, t] = segment.rewards[:, t] + gamma * next_return * is_terminal
        next_return = returns[:, t]

    advantages = returns - segment.values
    return advantages, returns


# @torch.no_grad()
# def basic_advantages_and_returns(
#     segment: TrajectorySegment, next_return: torch.Tensor, gamma=0.9
# ):
#     # Ensure the shape is correct and squeeze the rewards to have the shape [num_bots, rollout_size]
#     rewards = segment.rewards.squeeze()  # Squeeze any unnecessary dimensions
#     values = segment.values.squeeze()  # Similarly squeeze values to match the shape

#     # Initialize the returns tensor with the same shape as rewards
#     returns = torch.zeros_like(rewards).to(DEVICE).detach()

#     for t in reversed(
#         range(rewards.size(1))
#     ):  # Use the second dimension (rollout_size) for looping
#         if segment.dones[:, t].any():  # Check if any bot is done at this time step
#             returns[:, t] = rewards[:, t]
#         else:
#             returns[:, t] = rewards[:, t] + gamma * next_return

#         next_return = returns[:, t]

#     advantages = returns - values
#     return advantages, returns


class PPO:
    def __init__(
        self,
        env: TicTacToe,
        agent: AgentPPO,
        rollout_size=64,
        n_update_epochs=8,
        mini_batch_size=256,
        num_envs=32,
        # gae_enabled=True,
    ):
        self.env = env
        self.agent = agent.to(DEVICE)
        self.num_bots = num_envs
        # self.n_mini_batches = rollout_size // mini_batch_size
        self.n_update_epochs = n_update_epochs
        # self.gae_enabled = gae_enabled
        self.action_size = env.action_size
        self.state_size = env.get_state_size()
        self.rollout_size = rollout_size
        self.n_episode = 0
        self.mini_batch_size = mini_batch_size
        self.opps = [RandomAgent()]

    def add_opp(self, opp_filepath):
        opp_agent = AgentPPO(self.state_size, self.action_size)
        opp_agent.load(opp_filepath)
        opp_agent.eval()
        # opp_agent.set_eps = 0.1
        self.opps.append(opp_agent)

    def train(self, train_length, preload_filepath=None, continue_train_index=0):
        if preload_filepath is not None:
            self.agent.load(preload_filepath)
            self.opps.append(self.agent.clone())

        for i in range(
            continue_train_index + 1, continue_train_index + train_length + 1
        ):
            start_time = time.time()
            segment = self.collect_trajectory_segment()

            encode_state = self.env.get_encoded_states(segment.next_start_state)
            next_return = self.agent.get_value(encode_state).view(-1)
            advantages, returns = basic_advantages_and_returns(segment, next_return)

            batcher = Batcher(segment, advantages, returns, self.mini_batch_size)
            for _ in range(self.n_update_epochs):
                for mini_batch in batcher.shuffle():
                    self.agent.learn(mini_batch, 0.01)

            if i % 10 == 0:
                self.opps.append(self.agent.clone())

            if i % 50 == 0:
                self.agent.save(f"checkpoint/{i}.pt")

            end_time = time.time()

            print(f"Time taken for epoch {i}: {end_time - start_time:.4f} seconds")

    # def checkpoint_log(self, game_test_len=1000):
    #     random_agent = RandomAgent()
    #     result = 0
    #     for _ in range(game_test_len):
    #         result += self.simulate_game(self.agent, random_agent, self.env)
    #     print(f"Rewards {result} / {game_test_len}")
    #     return result / game_test_len

    # def simulate_game(agent: AgentPPO, opp: RandomAgent, env: TicTacToe):
    #     cur_player_index = np.random.randint(0, 2)
    #     state = env.get_initial_state()

    #     while True:
    #         cur_player_index = cur_player_index % 2
    #         if cur_player_index == 0:
    #             availalbe_action = env.get_valid_moves(state)
    #             action = agent.act(
    #                 env.get_encoded_single_state(state), availalbe_action
    #             )
    #             next_state = env.get_next_state(state, action)
    #             reward, is_terminated = env.get_value_and_terminated(next_state, action)

    #             if is_terminated:
    #                 return reward

    #         else:
    #             availalbe_action = env.get_valid_moves(state)
    #             action = opp.act(state, availalbe_action)
    #             next_state = env.get_next_state(state, action)
    #             reward, is_terminated = env.get_value_and_terminated(next_state, action)
    #             if is_terminated:
    #                 return -reward

    #         cur_player_index += 1

    #         state = env.reserve_state(next_state)

    def random_agent(self, opps: list[IAction]):
        if random.random() < 0.5:
            return opps[len(opps) - 1]
        else:
            return random.choice(opps)

    def collect_trajectory_segment(self):
        s_states = np.zeros(
            (
                self.num_bots,
                self.rollout_size,
                self.env.row_count,
                self.env.column_count,
            )
        )
        s_actions = np.zeros((self.num_bots, self.rollout_size))
        s_logprobs = np.zeros((self.num_bots, self.rollout_size))
        s_values = np.zeros((self.num_bots, self.rollout_size))
        s_rewards = np.zeros((self.num_bots, self.rollout_size))
        s_dones = np.zeros((self.num_bots, self.rollout_size))
        s_next_states = np.zeros(
            (
                self.num_bots,
                self.env.row_count,
                self.env.column_count,
            )
        )

        with ThreadPoolExecutor(max_workers=self.num_bots) as executor:
            futures = []
            for bot_num in range(self.num_bots):
                futures.append(
                    executor.submit(
                        self.simulate_trajectory,
                        s_states,
                        s_actions,
                        s_logprobs,
                        s_values,
                        s_rewards,
                        s_dones,
                        bot_num,
                    )
                )

            for i in range(len(futures)):
                s_next_states[i] = futures[i].result()

        return TrajectorySegment(
            torch.from_numpy(
                s_states.reshape(
                    (
                        self.num_bots,
                        self.rollout_size,
                        self.env.row_count * self.env.column_count,
                    )
                )
            )
            .to(DEVICE)
            .float(),
            torch.from_numpy(s_actions).to(DEVICE).float(),
            torch.from_numpy(s_logprobs).float(),
            torch.from_numpy(s_values).float(),
            torch.from_numpy(s_rewards).float(),
            torch.from_numpy(s_dones).float(),
            next_start_state=s_next_states,
        )

    def simulate_trajectory(
        self,
        s_states,
        s_actions,
        s_logprobs,
        s_values,
        s_rewards,
        s_dones,
        bot_num,
    ):
        start_state = self.env.get_initial_state()
        cur_state = start_state
        cur_player_index = 0
        agent_player_index = 0
        total_player = 2
        step = 0
        is_new_game = True
        opp = random.choice(self.opps)

        while step < self.rollout_size:
            if cur_player_index % total_player == agent_player_index:
                step, next_state, is_new_game = self.player_turn(
                    cur_state,
                    step,
                    s_states[bot_num],
                    s_actions[bot_num],
                    s_logprobs[bot_num],
                    s_values[bot_num],
                    s_rewards[bot_num],
                    s_dones[bot_num],
                )
            else:
                step, next_state, is_new_game = self.opp_turn(
                    cur_state,
                    opp,
                    step,
                    s_rewards[bot_num],
                    s_dones[bot_num],
                    is_new_game,
                )

            if is_new_game:
                opp = self.random_agent(self.opps)

            cur_state = self.env.reserve_state(next_state)
            cur_player_index += 1

        return cur_state

    def opp_turn(
        self,
        cur_state,
        opp: IAction,
        step,
        s_rewards,
        s_dones,
        is_new_game,
    ):
        encode_state = self.env.get_encoded_single_state(cur_state)
        action_mask = self.env.get_valid_moves(cur_state)
        action = opp.act(encode_state, action_mask)
        next_state = self.env.get_next_state(cur_state, action)
        # print(f"Opp play {action}")
        # print(f"Next state")
        # print(f"{next_state}")
        # print("------------------")

        reward, terminate = self.env.get_value_and_terminated(next_state, action)

        if not is_new_game:
            s_rewards[step] = 0
            s_dones[step] = terminate
            is_new_game = False
            if terminate:
                s_rewards[step] = self.modify_rewards_at_terminal(reward, False)
                next_state = self.env.get_initial_state()
                is_new_game = True
            return step + 1, next_state, is_new_game
        else:
            return step, next_state, False

    def modify_rewards_at_terminal(self, reward, is_agent):
        modify_reward = reward

        if reward == 1:
            if is_agent:
                modify_reward = 2
            else:
                modify_reward = -3

        if reward == 0:
            if is_agent:
                modify_reward = 1
            else:
                modify_reward = 1

        if reward == -1:
            if is_agent:
                modify_reward = -3
            else:
                modify_reward = 2

        return modify_reward

    def player_turn(
        self,
        cur_state,
        step,
        s_states,
        s_actions,
        s_logprobs,
        s_values,
        s_rewards,
        s_dones,
    ):
        encode_state = self.env.get_encoded_single_state(cur_state)
        action_mask = self.env.get_valid_moves(cur_state)
        action, logprob = self.agent.sample_action(encode_state, action_mask)
        action = action.item()
        value = self.agent.get_value(encode_state)

        next_state = self.env.get_next_state(cur_state, action)
        # print(f"Agent play {action}")
        # print(f"Next state")
        # print(f"{next_state}")
        # print("------------------")

        reward, terminate = self.env.get_value_and_terminated(next_state, action)

        if terminate:
            s_states[step] = cur_state
            s_actions[step] = action
            s_logprobs[step] = logprob
            s_values[step] = value
            s_rewards[step] = self.modify_rewards_at_terminal(reward, True)
            s_dones[step] = True
            return step + 1, self.env.get_initial_state(), True

        s_states[step] = cur_state
        s_actions[step] = action
        s_logprobs[step] = logprob
        s_values[step] = value
        return step, next_state, False


if __name__ == "__main__":
    game = TicTacToe()
    ppo = PPO(game, AgentPPO(game.get_state_size(), game.action_size))
    ppo = PPO(game, AgentPPO(game.get_state_size(), game.action_size))
    ppo.add_opp("checkpoints/v1.pt")
    ppo.add_opp("checkpoints/v2.pt")
    ppo.add_opp("checkpoints/v3.pt")
    ppo.train(200, "checkpoints/v4.pt")


# ppo.train(1000)

# opps = [RandomAgent()]

# ppo.collect_trajectory_segment(opps)
