import numpy as np
import optuna
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO.callbacks import AnnealEntropy, Report2Optuna, PrintAverageReward

from IndependentPPO.IPPO import IPPO
from IndependentPPO.subclasses import ParallelIPPO
from IndependentPPO.config import args_from_json
from hypertuning import OptunaOptimizeMultiObjective
import gym
import matplotlib


class OptimizerMOExample(OptunaOptimizeMultiObjective):
    def __init__(self, direction, study_name=None, save=None, n_trials=4):
        super().__init__(direction, study_name, save, n_trials)

    def objective(self, trial):
        args = {
            "verbose": False,
            "tb_log": True,
            "tag": "tiny",
            "env": "MultiAgentEthicalGathering-v1",
            "seed": 1,
            "max_steps": 500,
            "n_agents": 2,
            "n_steps": 2500,
            "tot_steps": 15000,
            "save_dir": "example_data",
            "early_stop": 15000000,
            "past_actions_memory": 0,
            "clip": 0.2,
            "target_kl": None,
            "gamma": 0.8,
            "gae_lambda": 0.95,
            "ent_coef": 0.04,
            "v_coef": 0.5,
            "actor_lr": 0.003,
            "critic_lr": 0.01,
            "anneal_lr": True,
            "n_epochs": 10,
            "norm_adv": True,
            "max_grad_norm": 1.0,
            "critic_times": 1,
            "h_size": 128,
            "last_n": 500,
            "n_cpus": 8,
            "th_deterministic": True,
            "cuda": False,
            "batch_size": 2500,
            "parallelize": True,
            "n_envs": 5,
            "h_layers": 2,
            "load": None,
            "clip_vloss": True,
            "anneal_entropy": True,
            "concavity_entropy": 1.8,
        }

        env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
        args["actor_lr"] = trial.suggest_float("actor_lr", 0.000005, 0.001)
        args["critic_lr"] = trial.suggest_float("critic_lr", 0.00005, 0.01)
        args["ent_coef"] = trial.suggest_float("ent_coef", 0.0001, 0.1)
        args["concavity-entropy"] = trial.suggest_float("concavity-entropy", 1.0, 3.5)
        ppo = ParallelIPPO(args, env=env)
        ppo.addCallbacks([
            PrintAverageReward(ppo, n=150),
            # TensorBoardLogging(ppo, log_dir="jro/EGG_DATA"),
            AnnealEntropy(ppo),
        ])

        ppo.train()
        metric = np.zeros(args["n_agents"])
        ppo.eval_mode = True
        for i in range(1):  # Sim does n_steps so keep it low
            rec = ppo.rollout()
            metric += rec
        metric /= 1
        return tuple(metric)


if __name__ == "__main__":
    optimizer = OptimizerMOExample(["maximize", "maximize"], save="example_data/optuna", study_name="test_MO")
    optimizer.optimize()
