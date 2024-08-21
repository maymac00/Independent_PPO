"""
Install MA ethical gathering game as test environment and its dependencies
pip install git+https://github.com/maymac00/MultiAgentEthicalGatheringGame.git
"""
from EthicalGatheringGame.presets import tiny, large
from EthicalGatheringGame.wrappers import NormalizeReward

from IndependentPPO import LIPPO
from IndependentPPO.subclasses import ParallelIPPO

from IndependentPPO.IPPO import IPPO

from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, AnnealActionFilter, PrintAverageRewardMO
from IndependentPPO.lr_schedules import IndependentPPOAnnealing
import gym

# matplotlib.use("TkAgg")
large["we"] = [1, 10]
large["objective_order"] = "individual_first"
large["color_by_efficiency"] = True
large["reward_mode"] = "vectorial"
env = gym.make("MultiAgentEthicalGathering-v1", **large)
args = {
    "verbose": False,
    "tb_log": True,
    "tag": "tiny",
    "env_name": "MultiAgentEthicalGathering-v1",
    "seed": 1,
    "max_steps": 500,
    "n_agents": 5,
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
    "h_size": 256,
    "last_n": 500,
    "n_cpus": 8,
    "th_deterministic": True,
    "cuda": False,
    "batch_size": 2500,
    "parallelize": True,
    "n_envs": 5,
    "h_layers": 3,
    "load": None,
    "anneal_entropy": True,
    "concavity_entropy": 1.8,
    "clip_vloss": True,
    "mult_lr": 0.01,
    "mult_init": 0.5,
    "constr_limit_1": 3,
    "constr_limit_2": 3,
    "log_gradients": False,
    "reward_size": 2,
    "beta_values": [2, 1],
    "eta_value": 0.1,
}

ppo = LIPPO(args, env=env)

ppo.lr_scheduler = IndependentPPOAnnealing(ppo, {
    0: {"actor_lr": 0.0001, "critic_lr": 0.01},
    1: {"actor_lr": 0.0001, "critic_lr": 0.01},
    2: {"actor_lr": 0.0001, "critic_lr": 0.01},
    3: {"actor_lr": 0.0001, "critic_lr": 0.01},
    4: {"actor_lr": 0.0001, "critic_lr": 0.01},
})
ppo.addCallbacks(PrintAverageRewardMO(ppo, 5, show_time=True))
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
