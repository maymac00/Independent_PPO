"""
Install MA ethical gathering game as test environment and its dependencies
pip install git+https://github.com/maymac00/MultiAgentEthicalGatheringGame.git
"""
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO.IPPO import IPPO
import gym
import matplotlib

matplotlib.use('TkAgg')

env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
args = {
    "verbose": False,
    "tb_log": True,
    "tag": "example",
    "env": "MultiAgentEthicalGathering-v1",
    "seed": 1,
    "max_steps": 500,
    "n_agents": 2,
    "n_steps": 2500,
    "tot_steps": 25000000,
    "save_dir": "example_data",
    "early_stop": 25000000.0,
    "past_actions_memory": 0,
    "clip": 0.2,
    "target_kl": None,
    "load": None,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "ent_coef": 0.2,
    "v_coef": 0.5,
    "actor_lr": 0.0003,
    "critic_lr": 0.005,
    "anneal_lr": True,
    "n_epochs": 10,
    "norm_adv": True,
    "max_grad_norm": 1.0,
    "critic_times": 2,
    "h_size": 256,
    "last_n": 500,
    "n_cpus": 8,
    "th_deterministic": True,
    "cuda": False,
    "batch_size": 2500,
    "parallelize": True,
    "n_envs": 5,
    "h_layers": 2,
}

ppo = IPPO(args, env=env)
ppo.train()

agents = IPPO.agents_from_file("IndependentPPO/example_data/example/2500_30000_1_ckpt")

# Run a simulation of the trained agents
obs, info = env.reset()
done = False
while not done:
    actions = [agent.predict(obs[i]) for i, agent in enumerate(agents)]
    obs, rewards, done, info = env.step(actions)
    env.render()
