import time
from abc import abstractmethod

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod


class Callback(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.ppo = None

    """
    Method to overload in case you need self.ppo for the constructor. This is needed since you do not have 
    access to qlearn instance on Callback.__init__"""

    def initiate(self):
        pass


class UpdateCallback(Callback):
    def __init__(self, ppo):
        self.ppo = ppo
        self.update_metrics = None

    @abstractmethod
    def after_update(self):
        pass

    @abstractmethod
    def before_update(self):
        pass


class LearningRateDecay(UpdateCallback):

    def __init__(self, ppo, decay=None, type="linear"):
        super().__init__(ppo)
        self.decay = decay
        self.type = type

    def after_update(self):
        if self.type == "linear":
            if self.decay is None:
                update = self.run_metrics["global_step"] / self.n_steps
                frac = 1.0 - (update - 1.0) / self.n_updates
                self.ppo.actor_lr = frac * self.init_args.actor_lr
                self.ppo.critic_lr = frac * self.init_args.critic_lr
            else:
                self.ppo.actor_lr *= self.decay
                self.ppo.critic_lr *= self.decay

    def before_update(self):
        pass


class AnnealEntropy(UpdateCallback):
    def __init__(self, ppo, base_value=1.0, final_value=0.1, concavity=3.5, type="linear_concave"):
        super().__init__(ppo)
        self.concavity = concavity
        self.base_value = base_value
        self.final_value = final_value
        self.type = type

    def before_update(self):
        pass

    def after_update(self):
        if self.type == "linear_concave":
            update = self.ppo.run_metrics["global_step"] / self.ppo.n_steps
            normalized_update = (update - 1.0) / self.ppo.n_updates
            complementary_update = 1 - normalized_update
            decay_step = normalized_update ** self.concavity / (
                    normalized_update ** self.concavity + complementary_update ** self.concavity)
            self.ppo.entropy_value = (self.base_value - self.final_value) * (1 - decay_step) + self.final_value
        elif self.type == "linear":
            update = self.ppo.run_metrics["global_step"] / self.ppo.n_steps
            frac = 1.0 - (update - 1.0) / self.ppo.n_updates
            self.ppo.entropy_value = frac * self.ppo.init_args.ent_coef


class TensorBoardLogging(UpdateCallback):
    def __init__(self, ppo, log_dir):
        super().__init__(ppo)
        self.writer = SummaryWriter(log_dir + f"/log/{self.ppo.init_args.tag}/{self.ppo.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(self.ppo.init_args).items()])),
        )

    def before_update(self):
        pass

    def after_update(self):
        # Log metrics from run metrics (avg reward), update metrics, and ppo parameters (e.g. entropy, lr)
        self.writer.add_scalar("Training/Avg Reward", np.array(self.ppo.run_metrics["avg_reward"]).mean(),
                               self.ppo.run_metrics["global_step"])
        self.writer.add_scalar("Training/Entropy coef", self.ppo.entropy_value, self.ppo.run_metrics["global_step"])
        self.writer.add_scalar("Training/Actor LR", self.ppo.actor_lr, self.ppo.run_metrics["global_step"])
        self.writer.add_scalar("Training/Critic LR", self.ppo.critic_lr, self.ppo.run_metrics["global_step"])
        self.writer.add_scalar("Training/SPS", (self.ppo.n_steps / self.ppo.run_metrics["sim_start_time"] - time.time()),
                               self.ppo.run_metrics["global_step"])
        for key, value in self.ppo.update_metrics.items():
            if isinstance(value, list):
                self.writer.add_scalar(key, np.array(value).mean(), self.ppo.run_metrics["global_step"])
            else:
                self.writer.add_scalar(key, value, self.ppo.run_metrics["global_step"])


# Printing Wrappers:
class PrintAverageReward(UpdateCallback):

    def __init__(self, ppo, n=100):
        super().__init__(ppo)
        self.n = n

    def after_update(self):
        if self.ppo.run_metrics["ep_count"] % self.n == 0:
            print(f"Average Reward: {np.array(self.ppo.run_metrics['avg_reward']).mean()}")

    def before_update(self):
        pass