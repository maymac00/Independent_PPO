import gym
from .IPPO import IPPO, _array_to_dict_tensor
import torch as th
import warnings
import numpy as np
from torch.multiprocessing import Manager, Pool
from .utils.memory import Buffer


class ParallelIPPO(IPPO):

    def __init__(self, args, env, run_name=None):
        super().__init__(args, env, run_name)
        if self.cuda:
            th.multiprocessing.set_start_method("spawn")

        self.logger.info("Parallelizing environments")
        self.parallelize = True
        self.available_threads = th.get_num_threads()
        main_process_threads = th.get_num_threads()
        if self.n_envs > main_process_threads:
            raise Exception(
                "Number of parallelized environments is greater than the number of available threads. Try with less environments or add more threads.")
        thld = main_process_threads - self.n_envs  # Number of threads to be used by the main process
        th.set_num_threads(thld)  # Set the number of threads to be used by the main process
        if self.n_envs < (self.n_steps / self.max_steps):
            warnings.warn(
                "Efficency is maximized when the number of parallelized environments is equal to n_steps/max_steps.")

    def rollout(self):

        with Manager() as manager:
            d = manager.dict()
            batch_size = int(self.n_steps / self.max_steps)

            solved = 0
            while solved < batch_size:
                runs = min(self.n_envs, batch_size - solved)
                # print("Running ", runs, " tasks")
                tasks = [(self.env, d, global_id, np.random.randint(0, 2**32 - 1)) for global_id in range(solved, solved + runs)]
                with Pool(runs) as p:
                    p.map(self._parallel_rollout, tasks[:runs])
                # Remove solved tasks
                solved += runs

            # Fetch the logs
            sim_metrics = self._parallel_results(d, batch_size)

            # We set the global env state of the environment to the first parallelized environment
            # This is important as it gets the wrappers info too.
            self.env.__dict__.update(d[0]["env_state"])
            # Merge the results
            for k in self.r_agents:
                buffs = [d[i]["single_buffer"][k] for i in range(batch_size)]
                self.buffer[k] = sum(buffs[1:], buffs[0])
        self.run_metrics['ep_count'] += solved
        self.run_metrics['global_step'] += solved * self.max_steps
        rew = np.array([s["reward_per_agent"] for s in sim_metrics])
        self.run_metrics['avg_reward'].append(rew.mean())
        # Save mean reward per agent
        for k in self.r_agents:
            self.run_metrics["agent_performance"][f"Agent_{k}/Reward"] = rew[:, k].mean()

        return np.array(
            [self.run_metrics["agent_performance"][f"Agent_{self.r_agents[k]}/Reward"] for k in self.r_agents])

    def _parallel_rollout(self, tasks):
        env, result, env_id, seed = tasks
        np.random.seed(seed)
        th.set_num_threads(1)
        data = {"global_step": self.run_metrics["global_step"], "reward_per_agent": None}

        single_buffer = {k: Buffer(self.o_size, self.max_steps, self.max_steps, self.gamma,
                                   self.gae_lambda, self.device) for k in self.r_agents}

        data["global_step"] += env_id * self.max_steps

        observation = self.environment_reset(env=env)


        action, logprob, s_value = [{k: 0 for k in self.r_agents} for _ in range(3)]
        env_action, ep_reward = [np.zeros(self.n_agents) for _ in range(2)]

        for step in range(self.max_steps):
            data["global_step"] += 1

            with th.no_grad():
                for k in self.r_agents:
                    (
                        env_action[k],
                        action[k],
                        logprob[k],
                        _,
                    ) = self.agents[k].actor.get_action(observation[k])

                    s_value[k] = self.agents[k].critic(observation[k])

            non_tensor_observation, reward, done, info = env.step(env_action)

            ep_reward += reward

            reward = _array_to_dict_tensor(self.r_agents, reward, self.device)
            done = _array_to_dict_tensor(self.r_agents, done, self.device)
            for k in self.r_agents:
                single_buffer[k].store(
                    observation[k],
                    action[k],
                    logprob[k],
                    reward[k],
                    s_value[k],
                    done[k]
                )

            observation = _array_to_dict_tensor(self.r_agents, non_tensor_observation, self.device)

        # End of simulation
        data["reward_per_agent"] = ep_reward
        data["single_buffer"] = single_buffer
        data["env_state"] = self.env.__dict__.items()
        result[env_id] = data

    def _parallel_results(self, d, batch_size):
        sim_metrics = [{}] * batch_size
        for i in range(batch_size):
            sim_metrics[i]["reward_per_agent"] = d[i]["reward_per_agent"]
            sim_metrics[i]["global_step"] = d[i]["global_step"]
        return sim_metrics

    def update(self):
        th.set_num_threads(self.available_threads)
        return super().update()


if __name__ == "__main__":
    from IndependentPPO.callbacks import AnnealEntropy, PrintAverageReward, TensorBoardLogging
    from EthicalGatheringGame.presets import tiny

    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)

    args = {
        "verbose": False,
        "tb_log": True,
        "tag": "tiny",
        "env_name": "MultiAgentEthicalGathering-v1",
        "seed": 1,
        "max_steps": 500,
        "n_agents": 2,
        "n_steps": 2500,
        "tot_steps": 15000,
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
        "concavity_entropy": 3.5,
        "clip_vloss": True,
        "log_gradients": False,
    }

    ppo = ParallelIPPO(args, env=env)
    ppo.addCallbacks(PrintAverageReward(ppo, 300))
    ppo.addCallbacks(AnnealEntropy(ppo, 1.0, 0.1, 3.5))
    ppo.addCallbacks(TensorBoardLogging(ppo, "example_data"))

    ppo.train()
