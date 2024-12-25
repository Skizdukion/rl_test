import random
import numpy as np
import torch
from agent import AgentPPO, IAction, RandomAgent
from env import TicTacToe
from replay_buffer import DEVICE


class PPO:
    def __init__(
        self,
        env: TicTacToe,
        agent: AgentPPO,
        rollout_size=1_024,
        # mini_batch_size=64,
        # n_update_epochs=4,
        # gae_enabled=True,
    ):
        self.env = env
        self.agent = agent.to(DEVICE)
        # self.num_bots = env.num_envs
        # self.n_mini_batches = rollout_size // mini_batch_size
        # self.n_update_epochs = n_update_epochs
        # self.gae_enabled = gae_enabled
        self.action_size = env.action_size
        self.state_size = env.get_state_size()
        self.rollout_size = rollout_size
        self.n_episode = 0

    def train(self):
        pass

    def collect_trajectory_segment(self, opps: IAction):
        start_state = self.env.get_initial_state()

        s_states = np.zeros(
            (self.rollout_size, self.env.row_count, self.env.column_count)
        )
        s_actions = np.zeros(self.rollout_size)
        s_logprobs = np.zeros(self.rollout_size)
        s_values = np.zeros(self.rollout_size)
        s_rewards = np.zeros(self.rollout_size)
        s_dones = np.zeros(self.rollout_size)

        cur_state = start_state
        cur_player_index = 0
        agent_player_index = 0
        total_player = 2
        step = 0
        is_new_game = True
        opp = random.choice(opps)

        while step < self.rollout_size:
            if cur_player_index % total_player == agent_player_index:
                step, next_state, is_new_game = self.player_turn(
                    cur_state,
                    step,
                    s_states,
                    s_actions,
                    s_logprobs,
                    s_values,
                    s_rewards,
                    s_dones,
                )
            else:
                step, next_state, is_new_game = self.opp_turn(
                    cur_state, opp, step, s_rewards, s_dones, is_new_game
                )

            if is_new_game:
                opp = random.choice(opps)

            cur_state = self.env.reserve_state(next_state)
            cur_player_index += 1

        return

    def opp_turn(
        self,
        cur_state,
        opp: IAction,
        step,
        s_rewards,
        s_dones,
        is_new_game,
    ):
        action_mask = self.env.get_valid_moves(cur_state)
        action = opp.act(cur_state, action_mask)
        next_state = self.env.get_next_state(cur_state, action)
        print(f"Opp play {action}")
        print(f"Next state {next_state}")
        reward, terminate = self.env.get_value_and_terminated(next_state, action)

        if not is_new_game:
            s_rewards[step] = -reward
            s_dones[step] = terminate
            is_new_game = False
            if terminate:
                next_state = self.env.get_initial_state()
                is_new_game = True
            return step + 1, next_state, is_new_game
        else:
            return step, next_state, False

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
        print(f"Agent play {action}")
        print(f"Next state {next_state}")

        reward, terminate = self.env.get_value_and_terminated(next_state, action)

        if terminate:
            s_states[step] = cur_state
            s_actions[step] = action
            s_logprobs[step] = logprob
            s_values[step] = value
            s_rewards[step] = reward
            s_dones[step] = True
            return step + 1, self.env.get_initial_state(), True

        s_states[step] = cur_state
        s_actions[step] = action
        s_logprobs[step] = logprob
        s_values[step] = value
        return step, next_state, False


game = TicTacToe()

ppo = PPO(game, AgentPPO(game.get_state_size(), game.action_size))

opps = [RandomAgent()]

ppo.collect_trajectory_segment(opps)
