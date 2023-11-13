import argparse
import time
import warnings

from IndependentPPO.utils.misc import *
import torch.nn as nn

import IndependentPPO


def _array_to_dict_tensor(agents: List[int], data: Array, device: th.device, astype: Type = th.float32) -> Dict:
    # Check if the provided device is already the current device
    is_same_device = (device == th.cuda.current_device()) if device.type == 'cuda' else (device == th.device('cpu'))

    if is_same_device:
        return {k: th.as_tensor(d, dtype=astype) for k, d in zip(agents, data)}
    else:
        return {k: th.as_tensor(d, dtype=astype).to(device) for k, d in zip(agents, data)}


class IPPO:
    def __init__(self, args, env, run_name=None):
        if type(args) is dict:
            args = argparse.Namespace(**args)
        elif type(args) is argparse.Namespace:
            args = args
        elif type(args) is str:
            args = IndependentPPO.config.args_from_json(args)
        self.init_args = args
        for k, v in args.__dict__.items():
            setattr(self, k, v)

        if run_name is not None:
            self.run_name = run_name
        else:
            # Get and format day and time
            timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
            self.run_name = f"{self.env}__{self.tag}__{self.seed}__{timestamp}__{np.random.randint(0, 100)}"
        print(f"Run name: {self.run_name}")

        # Action-Space
        self.o_size = None
        self.a_size = 7

        # Attributes
        self.agents = range(self.n_agents)
        self.run_metrics = {
            'global_step': 0,
            'ep_count': 0,
            'start_time': time.time(),
            'reward_q': [],
            'reward_per_agent': [],
            'avg_reward': [],
        }

        #   Actor-Critic
        self.n_updates = None
        self.buffer = None
        self.c_optim = None
        self.a_optim = None
        self.critic = {}
        self.actor = {}

        #   Torch init
        self.device = set_torch(self.n_cpus, self.cuda)
        self.env = env

        if self.parallelize:
            main_process_threads = th.get_num_threads()
            if self.n_envs > main_process_threads:
                raise Exception(
                    "Number of parallelized environments is greater than the number of available threads. Try with less environments or add more threads.")
            thld = main_process_threads - self.n_envs  # Number of threads to be used by the main process
            th.set_num_threads(thld)  # Set the number of threads to be used by the main process
            if self.n_envs < (self.n_steps / self.max_steps):
                warnings.warn(
                    "Efficency is maximized when the number of parallelized environments is equal to n_steps/max_steps.")

    def environment_reset(self, env=None):
        if env is None:
            non_tensor_observation, info = self.env.reset()
            observation = _array_to_dict_tensor(self.agents, non_tensor_observation, self.device)
            return observation
        else:
            non_tensor_observation, info = env.reset()
            observation = _array_to_dict_tensor(self.agents, non_tensor_observation, self.device)
            return observation

    def update(self):
        update_metrics = {}

        with th.no_grad():
            for k in self.agents:
                value_ = self.critic[k](self.environment_reset()[k])
                self.buffer[k].compute_mc(value_.reshape(-1))

        # Optimize the policy and value networks
        for k in self.agents:
            b = self.buffer[k].sample()
            self.buffer[k].clear()

            # Actor optimization
            for epoch in range(self.n_epochs):
                _, _, logprob, entropy = self.actor[k].get_action(b['observations'], b['actions'])
                entropy_loss = entropy.mean()

                update_metrics[f"Agent_{k}/Entropy"] = entropy_loss

                logratio = logprob - b['logprobs']
                ratio = logratio.exp()
                update_metrics[f"Agent_{k}/Ratio"] = ratio.mean()

                mb_advantages = b['advantages']
                if self.norm_adv: mb_advantages = normalize(mb_advantages)

                actor_loss = mb_advantages * ratio
                update_metrics[f"Agent_{k}/Non-Clipped Actor Loss"] = actor_loss.mean()

                actor_clip_loss = mb_advantages * th.clamp(ratio, 1 - self.clip, 1 + self.clip)
                # Calculate clip fraction
                actor_loss = th.min(actor_loss, actor_clip_loss).mean()
                update_metrics[f"Agent_{k}/Actor Loss"] = actor_loss

                actor_loss = -actor_loss - self.ent_coef * entropy_loss
                update_metrics[f"Agent_{k}/Actor Loss with Entropy"] = actor_loss

                self.a_optim[k].zero_grad(True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor[k].parameters(), self.max_grad_norm)
                self.a_optim[k].step()

            # Critic optimization
            for epoch in range(self.n_epochs * self.critic_times):
                values = self.critic[k](b['observations']).squeeze()

                critic_loss = 0.5 * ((values - b['returns']) ** 2).mean()
                update_metrics[f"Agent_{k}/Critic Loss"] = critic_loss

                critic_loss = critic_loss * self.v_coef
                update_metrics[f"Agent_{k}/Critic Loss with V Coef"] = critic_loss

                self.c_optim[k].zero_grad(True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic[k].parameters(), self.max_grad_norm)
                self.c_optim[k].step()

        return update_metrics

    def _sim(self):
        sim_metrics = {}

        observation = self.environment_reset()

        action, logprob, s_value = [{k: 0 for k in self.agents} for _ in range(3)]
        env_action, ep_reward = [np.zeros(self.n_agents) for _ in range(2)]

        for step in range(self.n_steps):
            self.run_metrics["global_step"] += 1

            with th.no_grad():
                for k in self.agents:
                    (
                        env_action[k],
                        action[k],
                        logprob[k],
                        _,
                    ) = self.actor[k].get_action(observation[k])
            # TODO: Change action -> env_action mapping
            non_tensor_observation, reward, done, info = self.env.step(env_action)
            ep_reward += reward

            reward = _array_to_dict_tensor(self.agents, reward, self.device)
            done = _array_to_dict_tensor(self.agents, done, self.device)
            for k in self.agents:
                self.buffer[k].store(
                    observation[k],
                    action[k],
                    logprob[k],
                    reward[k],
                    s_value[k],
                    done[k]
                )

            observation = _array_to_dict_tensor(self.agents, non_tensor_observation, self.device)

            # End of sim
            if all(list(done.values())):
                sim_metrics["reward_per_agent"] = ep_reward
                ep_reward = np.zeros(self.n_agents)
                # Reset environment
                observation = self.environment_reset()
        return sim_metrics

    def _parallel_sim(self, tasks):
