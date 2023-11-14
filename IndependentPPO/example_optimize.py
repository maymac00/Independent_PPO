from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO.IPPO import IPPO
from IndependentPPO.config import args_from_json
import gym
import matplotlib

args = {
    "verbose": False,
    "tb_log": True,
    "tag": "tiny",
    "env": "MultiAgentEthicalGathering-v1",
    "seed": 1,
    "max_steps": 500,
    "n_agents": 2,
    "n_steps": 2500,
    "tot_steps": 15000000,
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
    "load": None
}

def objective(trial):
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    args["tot_steps"] = 2500
    ppo = IPPO(args, env=env)
    ppo.train()
    metric = 0
    ppo.eval_mode = True
    for i in range(50):
        rec = ppo._sim()
        metric += sum([rec[f"Agent_{i}/Reward"] for i in range(args["n_agents"])])/args["n_agents"]
    metric /= 50

    return metric

if __name__ == "__main__":
    print(objective(None))