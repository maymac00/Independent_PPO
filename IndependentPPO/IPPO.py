import abc
import time
import PPO as ppo
import numpy as np
import torch as th
from PPO import PPOAgentExtraInfo, PPOAgent

from .callbacks import CallbackI, UpdateCallback


class IPPO(abc.ABC):

    def __init__(self, agents: list[ppo.PPOAgent] | list[ppo.PPOAgentExtraInfo], env, tot_steps, batch_size, **kwargs):
        self.agents = agents
        self.env = env
        self.total_steps = tot_steps
        self.batch_size = batch_size
        self.n_agents = len(agents)
        self.update_count = 0

        self.callbacks = []

        # Id of the run
        self.run_name = f"ippo_{time.time()}"

    @abc.abstractmethod
    def rollout(self):
        """
        Rollout the agents in the environment. Store transitions in the buffer.
        Returns
        -------
        """
        raise NotImplementedError

    def update_agent(self, agent: PPOAgentExtraInfo | PPOAgent, obs, **kwargs):
        """
        Update the agent. This can be overwridden for custom observations. For instance to add extra information to a cnn.
        Can also be parallelized.
        Parameters
        ----------
        agent : ppo.PPOAgent
            Agent to update.
        Returns
        -------
        """

        agent.update(obs)

    def train(self):
        """
        Train the agents.
        Returns
        -------
        """
        for update in range(1, self.total_steps // self.batch_size + 1):
            self.rollout()
            obs = self.env.reset()[0]

            for callback in self.callbacks:
                if isinstance(callback, UpdateCallback):
                    callback.before_update()
            # Update the agents
            for i, agent in enumerate(self.agents):
                self.update_agent(agent, obs[i])

            for callback in self.callbacks:
                if isinstance(callback, UpdateCallback):
                    callback.after_update()


    def save(self, path, save_critic=False):
        """
        Save the agents.
        Parameters
        ----------
        path : str
            Path to save the agents.
        Returns
        -------
        """
        for i, agent in enumerate(self.agents):
            agent.save(f"{path}/agent_{i}", save_critic)

    def add_callbacks(self, callbacks: CallbackI | list[CallbackI]):
        if isinstance(callbacks, list):
            self.callbacks.extend(callbacks)
        else:
            self.callbacks.append(callbacks)

