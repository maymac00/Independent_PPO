import PPO as ppo
import numpy as np
import torch as th
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import large
import gym
from PPO.callbacks import AnnealEntropyCallback
from torch import nn

from IndependentPPO.IPPO import IPPO
import warnings

from IndependentPPO.IPPO import IPPO
import warnings
from collections import deque

from IndependentPPO.ParallelIPPO import ParallelIPPO
from IndependentPPO.callbacks import TensorBoardCallback, PrintInfo

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    large["n_agents"] = 5
    large["we"] = [1, 3]
    large["efficiency"] = [1., 1., 1., 1., 1.]
    large["inequality_mode"] = "tie"
    large["obs_mode"] = "cnn"
    env = MAEGG(**large)
    # env = NormalizeReward(env)

    obs = env.reset(seed=0)[0][0]
    sample_obs = th.Tensor(obs['image'])
    recent_action = deque([4.0 / 6] * 5, maxlen=5)
    sample_extra_info = th.Tensor([obs['donation_box'], obs['survival_status'], *recent_action])

    # Train the agent
    total_steps = int(4e6)
    batch_size = 2500

    # Agent
    feature_map = [
        nn.Conv2d(1, 16, 3, 1, "same", bias=True),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
    ]

    feature_map = nn.ModuleList(feature_map)


    agents = []
    for i in range(5):
        ag = ppo.PPOAgent(
            ppo.actors.ConvSoftmaxActor(-1, 7, 128, 3, feature_map, sample_obs),
            ppo.critics.ConvCritic(-1, 64, 2, feature_map, sample_obs),
            ppo.Buffer(tuple(sample_obs.shape), batch_size, 250, 0.8, 0.95, th.device('cpu'))
        )

        ag = ppo.PPOAgentExtraInfo(ag, sample_extra_info.shape)
        ag.lr_scheduler = ppo.DefaultLrAnneal(ag, total_steps // batch_size)

        ag.addCallbacks([
            AnnealEntropyCallback(ag, total_steps // batch_size, 1.0, 0.2)
        ])
        agents.append(ag)


    class GatheringIPPO(IPPO):
        def get_obs_and_extra_info(self, state, recent_action):
            obs = [th.Tensor(ag_obs['image']) for ag_obs in state]
            extra_info = [
                th.Tensor([s['donation_box'], s['survival_status'], *past_actions])
                for s, past_actions in zip(state, recent_action)]
            return obs, extra_info

        def rollout(self):
            score = np.array([0.] * self.n_agents)
            state = env.reset(seed=0)[0]
            recent_action = [deque([4.0 / 6] * 5, maxlen=5)] * self.n_agents
            obs, extra_info = self.get_obs_and_extra_info(state, recent_action)
            eps = 0
            for step in range(batch_size):
                actions = []
                for i, ag in enumerate(self.agents):
                    action = ag.get_action(obs[i], cat=extra_info[i])
                    recent_action[i].append(action / 6.)
                    actions.append(action)

                state, reward, done, info = env.step(actions)

                obs, extra_info = self.get_obs_and_extra_info(state, recent_action)

                for i, agent in enumerate(self.agents):
                    agent.store_transition(reward[i], done[i])

                score += reward
                if all(done):
                    eps += 1
                    state = env.reset(seed=0)[0]
                    recent_action = [deque([4.0 / 6] * 5, maxlen=5)] * self.n_agents
                    obs, extra_info = self.get_obs_and_extra_info(state, recent_action)
            print(f"Mean Return: {score / eps}")

        def update_agent(self, agent: ppo.PPOAgentExtraInfo, obs, cat=None, **kwargs):
            recent_action = deque([4.0 / 6] * 5, maxlen=5)
            sample_extra_info = th.Tensor([obs['donation_box'], obs['survival_status'], *recent_action])
            agent.update(obs["image"], cat=sample_extra_info)

    class GatheringParallelIPPO(ParallelIPPO):
        def get_obs_and_extra_info(self, state, recent_action):
            obs = [th.Tensor(ag_obs['image']) for ag_obs in state]
            extra_info = [
                th.Tensor([s['donation_box'], s['survival_status'], *past_actions])
                for s, past_actions in zip(state, recent_action)]
            return obs, extra_info

        def _single_rollout(self, agents, env):
            score = np.array([0.] * self.n_agents)
            state = env.reset(seed=0)[0]
            recent_action = [deque([4.0 / 6] * 5, maxlen=5)] * self.n_agents
            obs, extra_info = self.get_obs_and_extra_info(state, recent_action)
            eps = 0
            for step in range(self.batch_size_for_worker):
                actions = []
                for i, ag in enumerate(self.agents):
                    action = ag.get_action(obs[i], cat=extra_info[i])
                    recent_action[i].append(action / 6.)
                    actions.append(action)

                state, reward, done, info = env.step(actions)

                obs, extra_info = self.get_obs_and_extra_info(state, recent_action)

                for i, agent in enumerate(self.agents):
                    agent.store_transition(reward[i], done[i])

                score += reward
                if all(done):
                    eps += 1

        def update_agent(self, agent: ppo.PPOAgentExtraInfo, obs, cat=None, **kwargs):
            recent_action = deque([4.0 / 6] * 5, maxlen=5)
            sample_extra_info = th.Tensor([obs['donation_box'], obs['survival_status'], *recent_action])
            agent.update(obs["image"], cat=sample_extra_info)



    # ippo = GatheringIPPO(agents, env, total_steps, batch_size)
    ippo = GatheringParallelIPPO(agents, env, total_steps, batch_size)
    ippo.add_callbacks([
        # TensorBoardCallback(ippo, "tb_cnn_example_data", 1),
        PrintInfo(ippo, 1)
    ])
    ippo.train()
    ippo.save(f"example_models/{ippo.run_name}")
