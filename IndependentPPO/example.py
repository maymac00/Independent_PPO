"""
Install MA ethical gathering game as test environment and its dependencies
pip install git+https://github.com/maymac00/MultiAgentEthicalGatheringGame.git
"""
from EthicalGatheringGame.presets import tiny
from EthicalGatheringGame.wrappers import NormalizeReward
from IPPO import IPPO
from LagrIPPO import LagrIPPO
from callbacks import AnnealEntropy, PrintAverageReward
from lr_schedules import  IndependentPPOAnnealing
import gym
import matplotlib

matplotlib.use("TkAgg")
tiny["we"] = [1, 99]
env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
env = NormalizeReward(env)
args = {
    "verbose": False,
    "tb_log": True,
    "tag": "tiny",
    "env_name": "MultiAgentEthicalGathering-v1",
    "seed": 1,
    "max_steps": 500,
    "n_agents": 2,
    "n_steps": 2500,
    "tot_steps": 5000000,
    "save_dir": "example_data",
    "early_stop": 15000,
    "past_actions_memory": 0,
    "clip": 0.2,
    "target_kl": None,
    "gamma": 0.8,
    "gae_lambda": 0.95,
    "ent_coef": 0.04,
    "v_coef": 0.5,
    "actor_lr": 0.0003,
    "critic_lr": 0.001,
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
    "anneal_entropy": True,
    "concavity_entropy": 1.8,
    "clip_vloss": True,
}

ppo = LagrIPPO(args, env=env)

ppo.lr_scheduler = IndependentPPOAnnealing(ppo, {
    0: {"actor_lr": 0.0003, "critic_lr": 0.004},
    1: {"actor_lr": 0.0001, "critic_lr": 0.0005},
})
ppo.addCallbacks(PrintAverageReward(ppo, 1))
ppo.addCallbacks(AnnealEntropy(ppo, 1.0, 0.5, args["concavity_entropy"]))
# ppo.addCallbacks(TensorBoardLogging(ppo, "example_data"))
# ppo.addCallbacks(SaveCheckpoint(ppo, 100))

import time

t0 = time.time()
ppo.train()
t = time.time() - t0
print(f"Steps per second: {ppo.tot_steps / t}")

agents = IPPO.actors_from_file("example_data/tiny/2500_5_1")

# Run a simulation of the trained agents
obs, info = env.reset()
done = False
while not done:
    actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
    obs, rewards, done, info = env.step(actions)
    done = all(done)
    env.render()
