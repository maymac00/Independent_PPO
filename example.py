"""
Install MA ethical gathering game as test environment and its dependencies
pip install git+https://github.com/maymac00/MultiAgentEthicalGatheringGame.git
"""
from EthicalGatheringGame.MultiAgentEthicalGathering import MAEGG
from IndependentPPO.IPPO import IPPO
import gym

env_config_dict = {
    'n_agents': 2,
    'map_size': 'large',
    'we': [1, 2.6],
    'inequality_mode': 'loss',
    'max_steps': 500,
    'apple_regen': 0.05,
    'donation_capacity': 10,
    'survival_threshold': 10,
    'visual_radius': 2,
    'partial_observability': True,
    'init_state': 'full'
}

env = gym.make("MultiAgentEthicalGathering-v1", **env_config_dict)
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
ppo = IPPO(args, env=env, env_params=env_config_dict)
ppo.train()
