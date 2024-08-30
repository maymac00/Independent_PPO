import PPO as ppo
import numpy as np
import torch as th
from EthicalGatheringGame.presets import tiny
import gym
from IndependentPPO.IPPO import IPPO
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    tiny["we"] = [1, 10]
    tiny["objective_order"] = "individual_first"
    tiny["color_by_efficiency"] = True
    tiny["reward_mode"] = "scalarised"
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    total_steps = int(3e6)
    batch_size = 5000

    agents = []
    for i in range(2):
        agents.append(ppo.PPOAgent(
            ppo.SoftmaxActor(env.observation_space.shape[0], 7, 256, 2),
            ppo.Critic(env.observation_space.shape[0], 256, 2),
            ppo.Buffer(env.observation_space.shape[0], batch_size, 500, 0.8, 0.95, th.device('cpu'))
        ))
        agents[-1].lr_scheduler = ppo.DefaultLrAnneal(agents[-1], total_steps // batch_size)


    class GatheringIPPO(IPPO):

        def rollout(self):
            obs = self.env.reset(seed=0)[0]
            score = np.array([0.] * self.n_agents)
            eps = 0
            for step in range(batch_size):
                actions = [agent.get_action(obs[k]) for k, agent in enumerate(self.agents)]
                state, reward, done, info = env.step(actions)

                for k, agent in enumerate(self.agents):
                    agent.store_transition(reward[k], done[k])

                score += reward
                if all(done):
                    eps += 1
                    obs = env.reset(seed=0)[0]
            print(f"Mean Return: {score / eps}")


    ippo = GatheringIPPO(agents, env, total_steps, batch_size)
    ippo.train()
    ippo.save(f"example_models/{ippo.run_name}")