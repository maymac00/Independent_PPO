"""Memory buffer script

This manages the memory buffer. 
"""
from copy import deepcopy

from .misc import *


@th.jit.script
def compute_gae(b_values: Tensor, value_: Tensor, b_rewards: Tensor, b_dones: Tensor, gamma: float, gae_lambda: float):
    values_ = th.cat((b_values[1:], value_))
    gamma = gamma * (1 - b_dones)
    deltas = b_rewards + gamma * values_ - b_values
    advantages = th.zeros_like(b_values)
    last_gaelambda = th.zeros_like(b_values[0])
    for t in range(advantages.shape[0] - 1, -1, -1):
        last_gaelambda = advantages[t] = deltas[t] + gamma[t] * gae_lambda * last_gaelambda

    returns = advantages + b_values

    return returns, advantages


class Buffer:
    """
    Class for the Buffer creation
    """

    def __init__(self, o_size: int, size: int, max_steps: int, gamma: float, gae_lambda: float, device: th.device):
        self.size = size

        # Assuming all episodes last for max_steps steps; otherwise fix sampling
        self.max_steps = max_steps

        # Take agents' observation space; discrete actions have size 1
        a_size = 1

        self.b_observations = th.zeros((self.size, o_size)).to(device)
        self.b_actions = th.zeros((self.size, a_size)).to(device)
        self.b_logprobs = th.zeros(self.size, dtype=th.float32).to(device)
        self.b_rewards = deepcopy(self.b_logprobs)
        self.b_values = deepcopy(self.b_logprobs)
        self.b_dones = deepcopy(self.b_logprobs)
        self.idx = 0

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.device = device

    def store(self, observation, action, logprob, reward, value, done):
        self.b_observations[self.idx] = observation
        self.b_actions[self.idx] = action
        self.b_logprobs[self.idx] = logprob
        self.b_rewards[self.idx] = reward
        self.b_values[self.idx] = value
        self.b_dones[self.idx] = done
        self.idx += 1

    def check_pos(self, idx):
        return self.b_observations[idx], self.b_actions[idx], self.b_logprobs[idx], self.b_rewards[idx], self.b_values[idx], self.b_dones[idx]

    # Store function prepared for parallelized simulations
    def store_parallel(self, idx, observation, action, logprob, reward, value, done):
        self.b_observations[idx] = observation
        self.b_actions[idx] = action
        self.b_logprobs[idx] = logprob
        self.b_rewards[idx] = reward
        self.b_values[idx] = value
        self.b_dones[idx] = done

    def compute_mc(self, value_):
        self.returns, self.advantages = compute_gae(self.b_values, value_, self.b_rewards, self.b_dones, self.gamma,
                                                    self.gae_lambda)

    def sample(self):
        n_episodes = int(self.size / self.max_steps)

        return {
            'observations': self.b_observations.reshape((n_episodes, self.max_steps, -1)),
            'actions': self.b_actions.reshape((n_episodes, self.max_steps, -1)),
            'logprobs': self.b_logprobs.reshape((n_episodes, self.max_steps)),
            'values': self.b_values.reshape((n_episodes, self.max_steps)),
            'returns': self.returns.reshape((n_episodes, self.max_steps)),
            'advantages': self.advantages.reshape((n_episodes, self.max_steps)),
        }

    def clear(self):
        self.idx = 0

    def detach(self):
        self.b_observations = self.b_observations.detach()
        self.b_actions = self.b_actions.detach()
        self.b_logprobs = self.b_logprobs.detach()
        self.b_rewards = self.b_rewards.detach()
        self.b_values = self.b_values.detach()
        self.b_dones = self.b_dones.detach()


def merge_buffers(buffers: list):
    """
    Merge multiple buffers into a single one.

    Args:
        buffers (list): List of buffers to merge.

    Returns:
        Buffer: Merged buffer.
    """
    # Get the device of the first buffer
    device = buffers[0].device

    # Instantiate the new merged buffer with the appropriate size
    o_size = buffers[0].b_observations.shape[-1]
    size = sum([buf.size for buf in buffers])
    max_steps = buffers[0].max_steps
    gamma = buffers[0].gamma
    gae_lambda = buffers[0].gae_lambda

    merged_buffer = Buffer(o_size, size, max_steps, gamma, gae_lambda, device)

    # Iterate through each buffer and concatenate its tensors
    merged_buffer.b_observations = th.cat([buf.b_observations for buf in buffers]).to(device)
    merged_buffer.b_actions = th.cat([buf.b_actions for buf in buffers]).to(device)
    merged_buffer.b_logprobs = th.cat([buf.b_logprobs for buf in buffers]).to(device)
    merged_buffer.b_rewards = th.cat([buf.b_rewards for buf in buffers]).to(device)
    merged_buffer.b_values = th.cat([buf.b_values for buf in buffers]).to(device)
    merged_buffer.b_dones = th.cat([buf.b_dones for buf in buffers]).to(device)

    # Handle the indices correctly
    merged_buffer.idx = size

    return merged_buffer
