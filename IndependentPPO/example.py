"""
Install MA ethical gathering game as test environment and its dependencies
pip install git+https://github.com/maymac00/MultiAgentEthicalGatheringGame.git
"""
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO.IPPO import IPPO
from IndependentPPO.config import args_from_json
import gym
import matplotlib

matplotlib.use('TkAgg')

env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
args = args_from_json("IndependentPPO/example_data/example/2500_30000_1_ckpt/config.json")
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
