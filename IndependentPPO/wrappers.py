import numpy as np


class PPOWrapper:
    def __init__(self, ppo):
        self.ppo = ppo

    def __getattr__(self, item):
        return getattr(self.ppo, item)


class LearningRateDecay(PPOWrapper):

    def __init__(self, ppo, decay):
        super().__init__(ppo)
        self.decay = decay

    def train(self):
        self.ppo.train()
        self.ppo.actor_lr *= self.decay
        self.ppo.critic_lr *= self.decay
