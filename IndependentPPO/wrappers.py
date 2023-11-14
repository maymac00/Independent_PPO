import numpy as np
from IndependentPPO.IPPO import IPPO


class PPOWrapper:
    def __init__(self, ppo):
        self.ppo = ppo

    def __getattr__(self, item):
        # If item is not in the wrapper, look for it in the wrapped object
        if item not in self.__dict__:
            return getattr(self.ppo, item)
        else:
            return self.__dict__[item]


class LearningRateDecay(PPOWrapper):

    def __init__(self, ppo, decay):
        super().__init__(ppo)
        self.decay = decay

    def train(self):
        self.ppo.train()
        self.ppo.actor_lr *= self.decay
        self.ppo.critic_lr *= self.decay


class AnnealEntropy(PPOWrapper):
    pass


# Printing Wrappers:
class PrintAverageReward():
    def __init__(self, ppo, n=100):
        self.ppo = ppo
        self.n = n

    def update(self):
        if self.ppo.run_metrics["ep_count"] % self.n == 0:
            print(f"Average Reward: {self.ppo.run_metrics['avg_reward'][-1]}")

    def __getattr__(self, item):
        # If item is not in the wrapper, look for it in the wrapped object
        if item not in self.__dict__:
            return getattr(self.ppo, item)
        else:
            return self.__dict__[item]
