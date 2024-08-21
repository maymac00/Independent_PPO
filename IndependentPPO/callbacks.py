import threading
import time
from abc import abstractmethod

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import sqlite3
import torch as th
from .ActionSelection import FilterSoftmaxActionSelection
from .agent import Agent, LagrAgent


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


class AnnealActionFilter(UpdateCallback):
    def __init__(self, ppo, thd_init=0.1):
        super().__init__(ppo)
        LagrAgent.action_filter = FilterSoftmaxActionSelection(range(self.ppo.a_size), threshold=thd_init)
        self.init_value = LagrAgent.action_filter.threshold

    def before_update(self):
        pass

    def after_update(self):
        if isinstance(LagrAgent.action_filter, FilterSoftmaxActionSelection):
            update = self.ppo.run_metrics["global_step"] / self.ppo.n_steps
            frac = (update - 1.0) / self.ppo.n_updates
            LagrAgent.action_filter.threshold = frac * self.init_value


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
    def __init__(self, ppo, log_dir, f=1, mo=False):
        super().__init__(ppo)
        self.mo = mo
        self.writer = SummaryWriter(log_dir=log_dir)
        self.freq = f  # Frequency of logging
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(self.ppo.init_args).items()])),
        )
        self.semaphore = threading.Semaphore(1)

    def before_update(self):
        pass

    def after_update(self):
        # TODO: Add a way to log the parameters of the agents individually
        with self.semaphore:
            if self.ppo.run_metrics["ep_count"] % self.freq == 0:
                th.set_num_threads(1)
                # Log metrics from run metrics (avg reward), update metrics, and ppo parameters (e.g. entropy, lr)
                if self.mo:
                    for r in range(self.ppo.reward_size):
                        self.writer.add_scalar(f"Training/Avg Reward Obj {r}",
                                       np.array(self.ppo.run_metrics[f"avg_reward_obj{r}"]).mean(),
                                       self.ppo.run_metrics["global_step"])
                else:
                    self.writer.add_scalar("Training/Avg Reward", np.array(self.ppo.run_metrics["avg_reward"]).mean(),
                                       self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/Entropy coef", self.ppo.entropy_value,
                                       self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/Actor LR", self.ppo.actor_lr, self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/Critic LR", self.ppo.critic_lr, self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/SPS",
                                       self.ppo.n_steps / (time.time() - self.ppo.run_metrics["sim_start_time"]),
                                       self.ppo.run_metrics["global_step"])
                self.writer.add_scalar("Training/Mean loss across agents",
                                       np.array(self.ppo.run_metrics["mean_loss"]).mean(),
                                       self.ppo.run_metrics["global_step"])

                for k, v in self.ppo.run_metrics["agent_performance"].items():
                    self.writer.add_scalar(k, v, self.ppo.run_metrics["global_step"])

                for key, value in self.ppo.update_metrics.items():
                    if isinstance(value, list):
                        self.writer.add_scalar(key, np.array(value).mean(), self.ppo.run_metrics["global_step"])
                    else:
                        self.writer.add_scalar(key, value, self.ppo.run_metrics["global_step"])


class Report2Optuna(UpdateCallback):
    def __init__(self, ppo, trial, n=1000, type="mean_reward"):
        super().__init__(ppo)
        self.trial = trial
        self.n = n
        self.type = type

    def after_update(self):
        if self.ppo.run_metrics["ep_count"] % self.n == 0:
            for i in range(5):
                try:
                    if self.type == "mean_reward":
                        self.trial.report(self.ppo.run_metrics["avg_reward"][-1], self.ppo.run_metrics["global_step"])
                    elif self.type == "mean_loss":
                        self.trial.report(abs(np.array(self.ppo.run_metrics["mean_loss"]).mean()),
                                          self.ppo.run_metrics["global_step"])
                    else:
                        raise NotImplementedError
                    break
                except sqlite3.OperationalError as e:
                    print(e)
                    print("Waiting 5 seconds to retry...")
                    time.sleep(5)
                except NotImplementedError:
                    raise NotImplementedError(f"Report type {self.type} not implemented.")

        if self.trial.should_prune():
            import optuna
            print(
                f"Trial {self.trial.number} pruned at step {self.ppo.run_metrics['global_step']} with value {self.ppo.run_metrics['avg_reward'][-1]}.")
            raise optuna.TrialPruned()

    def before_update(self):
        pass


# Printing Wrappers:
class PrintAverageReward(UpdateCallback):
    def __init__(self, ppo, n=100, show_time=False):
        super().__init__(ppo)
        self.n = n
        self.show_time = show_time
        self.t0 = time.time()

    def after_update(self):
        if self.ppo.run_metrics["ep_count"] % self.n == 0:
            s = ""
            s += f"Average Reward: {np.array(self.ppo.run_metrics['avg_reward']).mean()}"
            if self.show_time:
                s += f"\t | SPS: {self.ppo.max_steps * self.n / (time.time() - self.t0)}"
                self.t0 = time.time()
            print(s)

    def before_update(self):
        pass


class PrintAverageRewardMO(UpdateCallback):
    def __init__(self, ppo, n=100, show_time=False):
        super().__init__(ppo)
        self.n = n
        self.show_time = show_time
        self.t0 = time.time()

    def after_update(self):
        if self.ppo.run_metrics["ep_count"] % self.n == 0:
            s = ""
            for r in range(self.ppo.reward_size):
                s += f"Average Reward: {np.array(self.ppo.run_metrics[f'avg_reward_obj{r}']).mean()} \t"
            if self.show_time:
                s += f"\t | SPS: {self.ppo.max_steps * self.n / (time.time() - self.t0)}"
                self.t0 = time.time()
            print(s)

    def before_update(self):
        pass
