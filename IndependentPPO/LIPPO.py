import gym

from .IPPO import IPPO, _array_to_dict_tensor
import time
import torch as th
import torch.nn as nn
import warnings
import numpy as np
from torch.multiprocessing import Manager, Pool
from collections import deque

from .callbacks import UpdateCallback, PrintAverageRewardMO
from .utils.memory import LexicBuffer
from .agent import Agent, SoftmaxActor, LexicCritic
from .utils.misc import normalize, set_seeds


class LIPPO(IPPO):

    def __init__(self, args, env, run_name=None):
        super().__init__(args, env, run_name)

        #   Lexico params
        self.mu = [[0.0 for _ in range(self.reward_size - 1)] for _ in self.r_agents]
        self.j = [[0.0 for _ in range(self.reward_size - 1)] for _ in self.r_agents]
        self.recent_losses = [[deque(maxlen=50) for _ in range(self.reward_size)] for _ in self.r_agents]
        self.beta = [self.beta_values for _ in self.r_agents]
        self.eta = [[self.eta_value for _ in range(self.reward_size - 1)] for _ in self.r_agents]
        print(f"beta: {self.beta}")
        print(f"eta: {self.eta}")

        for k in self.r_agents:
            self.agents[k] = Agent(
                SoftmaxActor(self.o_size, self.a_size, self.h_size, self.h_layers).to(self.device),
                LexicCritic(self.o_size, self.reward_size, self.h_size, self.h_layers).to(self.device),
                self.actor_lr,
                self.critic_lr,
            )
            self.buffer[k] = LexicBuffer(self.o_size, self.reward_size, self.batch_size, self.max_steps, self.gamma,
                                         self.gae_lambda, self.device)

    def update(self):
        # Run callbacks
        for c in LIPPO.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.before_update()

        th.set_num_threads(self.n_cpus)
        update_metrics = {}

        with th.no_grad():
            for k in self.r_agents:
                value_ = self.agents[k].critic(self.environment_reset()[k])
                # tensor([-0.5803, -0.32423]), size 1
                self.buffer[k].compute_mc(value_.reshape(-1))

        # Optimize the policy and value networks
        for k in self.r_agents:
            self.buffer[k].clear()
            b = self.buffer[k].sample()

            # Actor optimization
            for epoch in range(self.n_epochs):
                first_order = []
                for i in range(self.reward_size - 1):  # remember that reward_size is 2
                    # it only enters one time, for i = 0
                    w = self.beta[k][i] + self.mu[k][i] * sum(self.beta[k][j] for j in range(i + 1, self.reward_size))
                    # computes only one weight, j takes value 1
                    first_order.append(w)
                first_order.append(self.beta[k][self.reward_size - 1])
                first_order_weights = th.tensor(first_order)

                _, _, logprob, entropy = self.agents[k].actor.get_action(b['observations'], b['actions'])
                # logprob and entropy have shape [5, 500].
                # 5 is the number of collected trajectories, and 500 the number of steps
                entropy_loss = entropy.mean()  # mean of all elements in the tensor. sum (all the values) / (5*500 elem)
                # entropy is now a scalar
                update_metrics[f"Agent_{k}/Entropy"] = entropy_loss.detach()

                logratio = logprob - b['logprobs']  # still size [5, 500]
                ratio = logratio.exp()  # still size [5, 500]
                update_metrics[f"Agent_{k}/Ratio"] = ratio.mean().detach()  # mean of all the values in the tensor

                mb_advantages = b['advantages']  # size [5, 500, 2]
                # Create the linear combination of normalized advantages
                first_order_weighted_advantages = first_order_weights[0] * normalize(mb_advantages[:, :, 0]) + \
                                                  first_order_weights[1] * normalize(mb_advantages[:, :, 1])  # [5, 500]
                actor_loss = first_order_weighted_advantages * ratio  # [5, 500]
                update_metrics[f"Agent_{k}/Actor Loss Non-Clipped"] = actor_loss.mean().detach()

                actor_clip_loss = first_order_weighted_advantages * th.clamp(ratio, 1 - self.clip, 1 + self.clip)

                # Log percent of clipped ratio
                update_metrics[f"Agent_{k}/Clipped Ratio"] = ((ratio < (1 - self.clip)).sum().item() + (
                            ratio > (1 + self.clip)).sum().item()) / np.prod(ratio.shape)

                # Calculate clip fraction
                actor_loss = th.min(actor_loss, actor_clip_loss).mean()
                update_metrics[f"Agent_{k}/Actor Loss"] = actor_loss.detach()

                actor_loss = -actor_loss - self.entropy_value * entropy_loss
                update_metrics[f"Agent_{k}/Actor Loss with Entropy"] = actor_loss.detach()

                # keep track of the recent losses from the first objective
                for i in range(self.reward_size):
                    # only compute the loss for the first reward
                    actor_loss_r0 = ratio * normalize(mb_advantages[:, :, i])
                    actor_clip_loss_r0 = normalize(mb_advantages[:, :, i]) * th.clamp(ratio, 1 - self.clip,
                                                                                      1 + self.clip)
                    actor_loss_r0 = th.min(actor_loss_r0, actor_clip_loss_r0).mean()
                    # after clipping, the loss is stored in the recent_losses list for evaluation in the lagrange mult
                    self.recent_losses[k][i].append(-actor_loss_r0 - self.entropy_value * entropy_loss)

                self.agents[k].a_optimizer.zero_grad(True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.agents[k].actor.parameters(), self.max_grad_norm)
                self.agents[k].a_optimizer.step()

            # Critic optimization
            for epoch in range(self.n_epochs * self.critic_times):
                values = self.agents[k].critic(b['observations']).squeeze()  # size [5, 500, 2]

                if self.clip_vloss:
                    v_loss_unclipped = (values - b['returns']) ** 2

                    v_clipped = (th.clamp(values, b['values'] - self.clip, b['values'] + self.clip) - b['returns']) ** 2
                    v_loss_clipped = th.min(v_loss_unclipped, v_clipped)

                    # Log percent of clipped ratio
                    update_metrics[f"Agent_{k}/Critic Clipped Ratio"] = ((values < (
                                b['values'] - self.clip)).sum().item() + (values > (b['values'] + self.clip)).sum().item()) / np.prod(values.shape)

                    critic_loss = 0.5 * v_loss_clipped.mean()
                    update_metrics[f"Agent_{k}/Critic Loss Non-Clipped"] = critic_loss.detach()
                else:
                    returns = b['returns']  # size [5, 500, 2]
                    critic_loss = nn.MSELoss()(values, returns)

                update_metrics[f"Agent_{k}/Critic Loss"] = critic_loss.detach()

                critic_loss = critic_loss * self.v_coef

                self.agents[k].c_optimizer.zero_grad(True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.agents[k].critic.parameters(), self.max_grad_norm)
                self.agents[k].c_optimizer.step()

            # Lagrange multipliers
            for i in range(self.reward_size - 1):  # only one iter, 0
                # only compute the mean of the last 25 losses
                self.j[k][i] = (-th.tensor(self.recent_losses[k][i])[25:]).mean()
                # update the lagrange multiplier -> just the mu[k][0] for the first reward is updated
            r = self.reward_size - 1
            for i in range(r):
                # difference between the last loss and the average of the last 25 losses weighted by the eta value
                self.mu[k][i] += self.eta[k][i] * (self.j[k][i] - (-self.recent_losses[k][i][-1]))
                # only keep it if it has worsened the loss, to give more weight to the first advantage
                self.mu[k][i] = max(0.0, self.mu[k][i])  # keep in mind we will only use mu[k][0] in the actor
            update_metrics[f"Agent_{k}/Mu_0"] = self.mu[k][0]

            loss = actor_loss + critic_loss
            update_metrics[f"Agent_{k}/Loss"] = loss.detach().cpu()
        self.update_metrics = update_metrics
        mean_loss = np.array([self.update_metrics[f"Agent_{k}/Loss"] for k in
                              self.r_agents]).mean()
        self.run_metrics["mean_loss"].append(mean_loss)

        # Run callbacks
        for c in LIPPO.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.after_update()

        return update_metrics

    def train(self, reset=True, set_agents=None):
        self.environment_setup()
        # set seed for training
        set_seeds(self.seed, self.th_deterministic)

        if reset:
            for k, v in self.init_args.__dict__.items():
                setattr(self, k, v)
        if set_agents is None:
            for k in self.r_agents:
                self.agents[k] = Agent(
                    SoftmaxActor(self.o_size, self.a_size, self.h_size, self.h_layers).to(self.device),
                    LexicCritic(self.o_size, self.reward_size, self.h_size, self.h_layers).to(self.device),
                    self.actor_lr,
                    self.critic_lr,
                )
                self.buffer[k] = LexicBuffer(self.o_size, self.reward_size, self.batch_size, self.max_steps, self.gamma,
                                             self.gae_lambda, self.device)
        else:
            self.agents = set_agents

        # Reset run metrics:
        self.run_metrics = {
            'global_step': 0,
            'ep_count': 0,
            'start_time': time.time(),
            'agent_performance': {},
            'mean_loss': deque(maxlen=500),
        }

        for r in range(self.reward_size):
            self.run_metrics[f'avg_reward_obj{r}'] = deque(maxlen=500)

        # Log relevant info before training
        self.logger.info(f"Training {self.run_name}")
        self.logger.info("-------------------TRAIN----------------")
        self.logger.info(f"Environment: {self.env}")
        self.logger.info(f"Number of agents: {self.n_agents}")
        self.logger.info(f"Number of steps: {self.n_steps}")
        self.logger.info(f"Total steps: {self.tot_steps}")
        self.logger.info(f"Number of hidden layers: {self.h_layers}")
        self.logger.info(f"Number of hidden units: {self.h_size}")
        self.logger.info("----------------------------------------")
        self.logger.info(f"Actor learning rate: {self.actor_lr}")
        self.logger.info(f"Critic learning rate: {self.critic_lr}")
        self.logger.info(f"Entropy coefficient: {self.ent_coef}")
        self.logger.info("-------------------CPV------------------")
        self.logger.info(f"Clip: {self.clip}")
        self.logger.info(f"Clip value loss: {self.clip_vloss}")
        self.logger.info("-------------------ENT------------------")
        self.logger.info(f"Anneal entropy: {self.anneal_entropy}")
        self.logger.info(f"Concavity entropy: {self.concavity_entropy}")
        self.logger.info("-------------------LRS------------------")
        # Log learning rate scheduler
        if self.lr_scheduler is not None:
            self.logger.info(f"Learning rate scheduler: {self.lr_scheduler}")
        else:
            self.logger.info("No learning rate scheduler")
        self.logger.info("----------------------------------------")
        self.logger.info(f"Seed: {self.seed}")

        # Training loop
        self.n_updates = self.tot_steps // self.batch_size
        for update in range(1, self.n_updates + 1):
            self.run_metrics["sim_start_time"] = time.time()

            self.rollout()

            self.update()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self._finish_training()

    def rollout(self):
        sim_metrics = {
            f"reward_per_agent_obj{k}": np.zeros(self.n_agents) for k in range(self.reward_size)
        }

        observation = self.environment_reset()

        action, logprob, s_value = [{k: 0 for k in self.r_agents} for _ in range(3)]
        env_action = np.zeros(self.n_agents)
        ep_reward = np.zeros(self.reward_shape)

        for step in range(self.n_steps):
            self.run_metrics["global_step"] += 1

            with th.no_grad():
                for k in self.r_agents:
                    (
                        env_action[k],
                        action[k],
                        logprob[k],
                        _,
                    ) = self.agents[k].actor.get_action(observation[k])
                    if not self.eval_mode:
                        s_value[k] = self.agents[k].critic(observation[k])
            # TODO: Change action -> env_action mapping
            non_tensor_observation, reward, done, info = self.env.step(env_action)
            ep_reward += reward

            reward = _array_to_dict_tensor(self.r_agents, reward, self.device)
            done = _array_to_dict_tensor(self.r_agents, done, self.device)
            if not self.eval_mode:
                for k in self.r_agents:
                    self.buffer[k].store(
                        observation[k],
                        action[k],
                        logprob[k],
                        reward[k],
                        s_value[k],
                        done[k]
                    )

            observation = _array_to_dict_tensor(self.r_agents, non_tensor_observation, self.device)

            # End of sim
            if all(list(done.values())):
                self.run_metrics["ep_count"] += 1
                for r in range(self.reward_size):
                    sim_metrics[f"reward_per_agent_obj{r}"] += ep_reward[:, r]
                ep_reward = np.zeros(self.reward_shape)
                # Reset environment
                observation = self.environment_reset()

        # sim_metrics["reward_per_agent"] /= (self.n_steps / self.max_steps)

        for r in range(self.reward_size):
            sim_metrics[f"reward_per_agent_obj{r}"] /= (self.n_steps / self.max_steps)
            self.run_metrics[f"avg_reward_obj{r}"].append(sim_metrics[f"reward_per_agent_obj{r}"].mean())

        # Save mean reward per agent
        for k in self.r_agents:
            for r in range(self.reward_size):
                self.run_metrics["agent_performance"][f"Agent_{k}/Reward Obj {r}"] = \
                    sim_metrics[f"reward_per_agent_obj{r}"][k].mean()
        return np.array(
            [self.run_metrics["agent_performance"][f"Agent_{self.r_agents[k]}/Reward Obj {r}"] for k in self.r_agents])

    def _finish_training(self):
        # Log relevant data from training
        self.logger.info(f"Training finished in {time.time() - self.run_metrics['start_time']} seconds")
        for r in range(self.reward_size):
            self.logger.info(f"Average reward obj {r}: {np.mean(self.run_metrics[f'avg_reward_obj{r}'])}")
        self.logger.info(f"Average loss: {np.mean(self.run_metrics['mean_loss'])}")
        self.logger.info(f"Std mean loss: {np.std(self.run_metrics['mean_loss'])}")
        self.logger.info(f"Number of episodes: {self.run_metrics['ep_count']}")
        self.logger.info(f"Number of updates: {self.n_updates}")

        self.save_experiment_data()


class ParallelLIPPO(LIPPO):
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
            # self.env.__dict__.update(d[0]["env_state"])
            self.environment_reset()
            # Merge the results
            for k in self.r_agents:
                buffs = [d[i]["single_buffer"][k] for i in range(batch_size)]
                self.buffer[k] = sum(buffs[1:], buffs[0])
        self.run_metrics['ep_count'] += solved
        self.run_metrics['global_step'] += solved * self.max_steps

        for r in range(self.reward_size):
            self.run_metrics[f'avg_reward_obj{r}'].append(
                np.array([s[f"reward_per_agent_obj{r}"] for s in sim_metrics]).mean())

        # Save mean reward per agent
        for k in self.r_agents:
            for r in range(self.reward_size):
                self.run_metrics["agent_performance"][f"Agent_{k}/Reward Obj {r}"] = \
                    sim_metrics[k][f"reward_per_agent_obj{r}"].mean()
        return np.array(
            [self.run_metrics["agent_performance"][f"Agent_{self.r_agents[k]}/Reward Obj {r}"] for k in
             self.r_agents])

    def _parallel_rollout(self, tasks):
        env, result, env_id, seed= tasks
        np.random.seed(seed)
        th.set_num_threads(1)

        data = {
            f"reward_per_agent_obj{k}": np.zeros(self.n_agents) for k in range(self.reward_size)
        }
        data["global_step"] = self.run_metrics["global_step"]

        single_buffer = {k: LexicBuffer(self.o_size, self.reward_size, self.max_steps, self.max_steps, self.gamma,
                                        self.gae_lambda, self.device) for k in self.r_agents}

        data["global_step"] += env_id * self.max_steps

        observation = self.environment_reset(env=env)

        action, logprob, s_value = [{k: 0 for k in self.r_agents} for _ in range(3)]

        env_action = np.zeros(self.n_agents)
        ep_reward = np.zeros(self.reward_shape)

        for step in range(self.max_steps):
            self.run_metrics["global_step"] += 1

            with th.no_grad():
                for k in self.r_agents:
                    (
                        env_action[k],
                        action[k],
                        logprob[k],
                        _,
                    ) = self.agents[k].actor.get_action(observation[k])
                    if not self.eval_mode:
                        s_value[k] = self.agents[k].critic(observation[k])
            # TODO: Change action -> env_action mapping
            non_tensor_observation, reward, done, info = self.env.step(env_action)
            ep_reward += reward

            reward = _array_to_dict_tensor(self.r_agents, reward, self.device)
            done = _array_to_dict_tensor(self.r_agents, done, self.device)
            if not self.eval_mode:
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
        for r in range(self.reward_size):
            data[f"reward_per_agent_obj{r}"] += ep_reward[:, r]
        data["single_buffer"] = single_buffer
        #data["env_state"] = copy.deepcopy(self.env.__dict__.items())
        result[env_id] = data

    def _parallel_results(self, d, batch_size):
        sim_metrics = [{}] * batch_size
        for i in range(batch_size):
            for r in range(self.reward_size):
                sim_metrics[i][f"reward_per_agent_obj{r}"] = d[i][f"reward_per_agent_obj{r}"]
            sim_metrics[i]["global_step"] = d[i]["global_step"]
        return sim_metrics

    def update(self):
        th.set_num_threads(self.available_threads)
        return super().update()


if __name__ == "__main__":
    from callbacks import AnnealEntropy, PrintAverageReward, TensorBoardLogging
    from lr_schedules import DefaultPPOAnnealing
    from EthicalGatheringGame.presets import tiny

    tiny["reward_mode"] = "vectorial"
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
        "tot_steps": 1500000,
        "save_dir": "example_data",
        "early_stop": 15000,
        "past_actions_memory": 0,
        "clip": 0.2,
        "target_kl": None,
        "gamma": 0.8,
        "gae_lambda": 0.95,
        "ent_coef": 0.04,
        "v_coef": 0.5,
        "actor_lr": 0.0015,
        "critic_lr": 0.008,
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
        "reward_size": 2,
        "beta_values": [2, 1],
        "eta_value": 0.1,
    }

    ppo = LIPPO(args, env=env)
    ppo.lr_scheduler = DefaultPPOAnnealing(ppo)
    ppo.addCallbacks([
        PrintAverageRewardMO(ppo, 5, show_time=True),
        TensorBoardLogging(ppo, log_dir=f"{args['save_dir']}/{args['tag']}/log/{ppo.run_name}", f=1, mo=True),
        AnnealEntropy(ppo, 1.0, 0.4, 3.5),])
    ppo.train()
