import time
from abc import abstractmethod

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import sqlite3


class Callback(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.ppo = None

    def initiate(self):
        """
           Method to overload in case you need self.ppo for the constructor. This is needed since you do not have
           access to qlearn instance on Callback.__init__"""
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
            frac = (self.base_value - self.final_value) * (1 - decay_step) + self.final_value
            self.ppo.entropy_value = frac * self.ppo.init_args.ent_coef
        elif self.type == "linear":
            update = self.ppo.run_metrics["global_step"] / self.ppo.n_steps
            frac = 1.0 - (update - 1.0) / self.ppo.n_updates
            self.ppo.entropy_value = frac * self.ppo.init_args.ent_coef


class SaveCheckpoint(UpdateCallback):
    def __init__(self, ppo, n=1000):
        super().__init__(ppo)
        self.n = n

    def before_update(self):
        pass

    def after_update(self):
        if self.ppo.run_metrics["ep_count"] % self.n == 0:
            self.ppo.save_experiment_data(ckpt=True)


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
        self.writer.add_scalar("Training/SPS",
                               self.ppo.n_steps / (time.time() - self.ppo.run_metrics["sim_start_time"]),
                               self.ppo.run_metrics["global_step"])
        self.writer.add_scalar("Training/Loss",
                               self.ppo.run_metrics["Mean loss across agents"],
                               self.ppo.run_metrics["global_step"])

        for k, v in self.ppo.run_metrics["agent_performance"].items():
            self.writer.add_scalar(k, v, self.ppo.run_metrics["global_step"])

        for key, value in self.ppo.update_metrics.items():
            if isinstance(value, list):
                self.writer.add_scalar(key, np.array(value).mean(), self.ppo.run_metrics["global_step"])
            else:
                self.writer.add_scalar(key, value, self.ppo.run_metrics["global_step"])


class Report2Optuna(UpdateCallback):
    def __init__(self, ppo, trial, n=1000):
        super().__init__(ppo)
        self.trial = trial
        self.n = n

    def after_update(self):
        if self.ppo.run_metrics["ep_count"] % self.n == 0:
            for i in range(5):
                try:
                    self.trial.report(self.ppo.run_metrics["avg_reward"][-1], self.ppo.run_metrics["global_step"])
                    break
                except sqlite3.OperationalError as e:
                    print(e)
                    print("Waiting 5 seconds to retry...")
                    time.sleep(5)

        if self.trial.should_prune():
            import optuna
            print(
                f"Trial {self.trial.number} pruned at step {self.ppo.run_metrics['global_step']} with value {self.ppo.run_metrics['avg_reward'][-1]}.")
            raise optuna.TrialPruned()

    def before_update(self):
        pass


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
