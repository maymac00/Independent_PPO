import time
from collections import deque
from torch.multiprocessing import Process, Manager
import torch.multiprocessing as mp
import logging
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import CommonsGame.envs.env
from agent import SoftmaxActor, Critic, ACTIONS
from utils.memory import Buffer, merge_buffers
from utils.misc import *
import config
import gym
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def is_pickleable(obj):
    try:
        pickle.loads(pickle.dumps(obj))
        return True
    except Exception as e:
        print(e)
        return False


def _array_to_dict_tensor(agents: List[int], data: Array, device: th.device, astype: Type = th.float32) -> Dict:
    # Check if the provided device is already the current device
    is_same_device = (device == th.cuda.current_device()) if device.type == 'cuda' else (device == th.device('cpu'))

    if is_same_device:
        return {k: th.as_tensor(d, dtype=astype) for k, d in zip(agents, data)}
    else:
        return {k: th.as_tensor(d, dtype=astype).to(device) for k, d in zip(agents, data)}


class IPPO:
    summary_w = None

    def __init__(self, args, run_name=None, env=None):
        self.args = args
        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = f"{args.env}__{args.tag}__{args.seed}__{int(time.time())}__{np.random.randint(0, 100)}"
        self.eval_mode = False

        # Action-Space
        self.o_size = None
        self.a_size = 7

        # Attributes
        self.agents = range(args.n_agents)
        self.metrics = {
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
        self.past_actions_memory = {}

        #   Torch init
        self.device = set_torch(self.args.n_cpus, self.args.cuda)
        self.env = env

        if self.args.parallelize:
            main_process_threads = th.get_num_threads()
            print("Main process threads: ", th.get_num_threads())
            print("Number of parallelized environments: ", self.args.n_envs)
            if self.args.n_envs > main_process_threads:
                raise Exception(
                    "Number of parallelized environments is greater than the number of available threads. Try with less environments or add more threads.")
            thld = main_process_threads - self.args.n_envs  # Number of threads to be used by the main process
            th.set_num_threads(thld)  # Set the number of threads to be used by the main process
            if self.args.n_envs < (self.args.n_steps / self.args.max_steps):
                warnings.warn("Efficency is maximized when the number of parallelized environments is equal to n_steps/max_steps.")

            mp.set_start_method('spawn')

    def environment_setup(self):
        if self.env is None:
            raise Exception("Environment not set")
        obs = self.env.reset()
        # Decentralized learning checks
        if isinstance(obs, list):
            if len(obs) != self.args.n_agents:
                raise Exception("The environment returns a list of observations but the number of agents "
                                "is not the same as the number of observations.")
        elif isinstance(obs, np.ndarray):
            if len(obs.shape) != 2:
                raise Exception("The environment returns a numpy array of observations but the shape is not 2D. It "
                                "should be (agents x observation).")
        else:
            raise Exception("Observation is not a list neither an array.")

        self.o_size = len(self.env.reset()[0]) + self.args.past_actions_memory
        self.a_size = 7
        print(f"Observation space: {self.o_size}, Action space: {self.a_size}")
        # TODO: Set the action space, translate actions to env_actions

    def environment_reset(self, env=None):
        if env is None:
            non_tensor_observation = self.env.reset()
            if self.args.past_actions_memory > 0:
                # TODO: Reset intial memory for the previous actions taken
                pass
            observation = _array_to_dict_tensor(self.agents, non_tensor_observation, self.device)
            return observation
        else:
            non_tensor_observation = env.reset()
            if self.args.past_actions_memory > 0:
                # TODO: Reset intial memory for the previous actions taken
                pass
            observation = _array_to_dict_tensor(self.agents, non_tensor_observation, self.device)
            return observation

    def train(self, load_from_checkpoint=None, init_global_step=0):
        self.environment_setup()
        # set seed for training
        set_seeds(args.seed, self.args.th_deterministic)
        IPPO.summary_w, self.wandb_path = init_loggers(self.run_name, self.args)

        # Init actor-critic setup
        self.actor, self.critic, self.a_optim, self.c_optim, self.buffer = {}, {}, {}, {}, {}
        self.past_actions_memory = {}
        # TODO: Reset intial memory for the previous actions taken
        # initial_memory = [ACTIONS.index(CommonsGame.STAY) / len(ACTIONS) for i in range(self.args.past_actions_memory)]

        if load_from_checkpoint is not None:
            self._load_models_from_files(load_from_checkpoint)
            for k in self.agents:
                # self.past_actions_memory[k] = deque(initial_memory, maxlen=self.args.past_actions_memory)
                self.a_optim[k] = optim.Adam(list(self.actor[k].parameters()), lr=self.args.actor_lr, eps=1e-5)
                self.c_optim[k] = optim.Adam(list(self.critic[k].parameters()), lr=self.args.critic_lr, eps=1e-5)
                self.buffer[k] = Buffer(self.o_size, self.args.n_steps, self.args.max_steps, self.args.gamma,
                                        self.args.gae_lambda, self.device)
        else:
            for k in self.agents:
                self.actor[k] = SoftmaxActor(self.o_size, self.a_size, self.args.h_size, self.args.h_layers).to(
                    self.device)
                self.a_optim[k] = optim.Adam(list(self.actor[k].parameters()), lr=self.args.actor_lr, eps=1e-5)
                self.critic[k] = Critic(self.o_size, self.args.h_size, self.args.h_layers).to(self.device)
                self.c_optim[k] = optim.Adam(list(self.critic[k].parameters()), lr=self.args.critic_lr, eps=1e-5)
                # self.past_actions_memory[k] = deque(initial_memory, maxlen=self.args.past_actions_memory)
                self.buffer[k] = Buffer(self.o_size, self.args.n_steps, self.args.max_steps, self.args.gamma,
                                        self.args.gae_lambda, self.device)

        # Reset Training metrics
        self.metrics = {
            'global_step': init_global_step,
            'ep_count': init_global_step / args.max_steps,
            'start_time': time.time(),
            'reward_q': [],
            'reward_per_agent': [],
            'avg_reward': deque(maxlen=500),
        }

        # Training loop
        self.n_updates = self.args.tot_steps // self.args.batch_size
        for update in range(1, self.n_updates + 1):
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.n_updates
                for a_opt, c_opt in zip(self.a_optim.values(), self.c_optim.values()):
                    a_opt.param_groups[0]["lr"] = frac * self.args.actor_lr
                    c_opt.param_groups[0]["lr"] = frac * self.args.critic_lr
                if self.args.tb_log: IPPO.summary_w.add_scalar('Training/Actor LR', a_opt.param_groups[0]["lr"],
                                                               self.metrics["global_step"])
                if self.args.tb_log: IPPO.summary_w.add_scalar('Training/Critic LR', c_opt.param_groups[0]["lr"],

                                                               self.metrics["global_step"])

            if not self.args.parallelize:
                sim_start = time.time()
                self._sim()
                print(f"Sim time: {time.time() - sim_start}")
            else:
                with Manager() as manager:
                    d = manager.dict()

                    batch_start_time = time.time()
                    batch_size = int(self.args.n_steps / self.args.max_steps)

                    tasks = [(self.env, d, i) for i in range(batch_size)]
                    solved = 0
                    while solved < batch_size:
                        runs = min(self.args.n_envs, batch_size - solved)
                        print("Running ", runs, " tasks")
                        with mp.Pool(runs) as p:
                            p.map(self._parallel_sim, tasks[:runs])
                        # Remove solved tasks
                        solved += runs
                        tasks = tasks[runs:]

                    print("Batch time: ", time.time() - batch_start_time)
                    # Fetch the logs
                    for i in range(batch_size):
                        for tup in d[i]["logs"]:
                            k, v, x = tup
                            IPPO.summary_w.add_scalar(k, v, x)

                    # Merge the results
                    # TODO: Merge Buffers
                    for k in self.agents:
                        self.buffer[k] = merge_buffers([d[i]["single_buffer"][k] for i in range(self.args.n_envs)])
                    self.metrics['ep_count'] += self.args.n_envs
                    self.metrics['global_step'] += self.args.n_envs * self.args.max_steps
                    self.metrics['reward_q'] += [res["reward_q"] for res in d.values()]
                    self.metrics['reward_per_agent'] += [res["reward_per_agent"] for res in d.values()]
                    self.metrics['avg_reward'].appendleft(sum(self.metrics['reward_q']) / len(self.metrics['reward_q']))


                if self.args.verbose:
                    print(f"E: {self.metrics['ep_count']},\n\t "
                          f"Avg_Reward for all episodes: {np.mean(self.metrics['avg_reward'])},\n\t"
                          f"Global_Step: {self.metrics['global_step']},\n\t "
                          )
            self._update()

            # TODO: Callbacks
            sps = int(self.metrics["global_step"] / (time.time() - self.metrics["start_time"]))
            if self.args.tb_log: self.summary_w.add_scalar('Training/SPS', sps, self.metrics["global_step"])

    def _load_models_from_files(self, load_from_checkpoint):
        pass

    def _update(self):
        with th.no_grad():
            for k in self.agents:
                value_ = self.critic[k](self.environment_reset()[k])
                self.buffer[k].compute_mc(value_.reshape(-1))

        # Optimize the policy and value networks
        for k in self.agents:
            b = self.buffer[k].sample()
            self.buffer[k].clear()
            # Actor optimization
            for epoch in range(self.args.n_epochs):
                _, _, logprob, entropy = self.actor[k].get_action(b['observations'], b['actions'])
                entropy_loss = entropy.mean()

                logratio = logprob - b['logprobs']
                ratio = logratio.exp()

                mb_advantages = b['advantages']
                if self.args.norm_adv: mb_advantages = normalize(mb_advantages)

                actor_loss = mb_advantages * ratio

                actor_clip_loss = mb_advantages * th.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip)
                actor_loss = th.min(actor_loss, actor_clip_loss).mean()
                if self.args.tb_log: IPPO.summary_w.add_scalar('Agent_' + str(k) + "/" + 'Actor loss', actor_loss,
                                                               (self.metrics[
                                                                    "global_step"] / self.args.max_steps) * self.args.n_epochs + epoch)

                actor_loss = -actor_loss - self.args.ent_coef * entropy_loss
                # if args.tb_log: summary_w.add_scalar('Agent_' + str(k) + "/" + 'Actor loss', actor_loss,
                # (global_step/args.max_steps)*args.n_epochs+epoch)
                self.a_optim[k].zero_grad(True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor[k].parameters(), self.args.max_grad_norm)
                self.a_optim[k].step()

                # Not using this rn
                """
                with th.no_grad():  # Early break from updates
                    if args.target_kl is not None:
                        approx_kl = ((ratio - 1) - logratio).mean()
                        if approx_kl > args.target_kl:
                            break"""

            # Critic optimization
            for epoch in range(self.args.n_epochs * self.args.critic_times):
                values = self.critic[k](b['observations']).squeeze()

                critic_loss = 0.5 * ((values - b['returns']) ** 2).mean()

                if self.args.tb_log: IPPO.summary_w.add_scalar('Agent_' + str(k) + "/" + 'Critic loss', critic_loss,
                                                               (self.metrics[
                                                                    "global_step"] / self.args.max_steps) * self.args.n_epochs * self.args.critic_times + epoch)

                critic_loss = critic_loss * self.args.v_coef

                self.c_optim[k].zero_grad(True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic[k].parameters(), self.args.max_grad_norm)
                self.c_optim[k].step()

    def _sim(self, render=False, masked=False, pause=0.01):
        if self.eval_mode:
            self.args.tb_log = False

        observation = self.environment_reset()
        self.last_run = {
            "greedy": 0,
            "reward_per_agent": []
        }
        info = {
            "donationBox": 0,
            "n": [0] * self.args.n_agents,
            "donationBox_full": False,
        }

        action, logprob, s_value = [{k: 0 for k in self.agents} for _ in range(3)]
        env_action, ep_reward = [np.zeros(self.args.n_agents) for _ in range(2)]
        job_done = False
        apple_history = []
        donation_box_history = []
        for step in range(self.args.n_steps):
            self.metrics["global_step"] += 1  # * args.n_envs

            if render:
                self.env.render(masked=masked)
                time.sleep(pause)

            with th.no_grad():
                for k in self.agents:
                    (
                        env_action[k],
                        action[k],
                        logprob[k],
                        _,
                    ) = self.actor[k].get_action(observation[k])
                    if self.args.past_actions_memory > 0:
                        self.past_actions_memory[k].appendleft(float(action[k] / len(ACTIONS)))
                    if not self.eval_mode:
                        s_value[k] = self.critic[k](observation[k])

            non_tensor_observation, reward, done, info = self.env.step(env_action)
            if self.args.past_actions_memory > 0:
                for k in self.agents:
                    non_tensor_observation[k] = np.append(non_tensor_observation[k], self.past_actions_memory[k])
            apple_history.append(info['n'])
            donation_box_history.append(info['donationBox'])

            if self.metrics["global_step"] % self.args.max_steps == 0:
                done = [True] * self.args.n_agents

            # Consider the metrics of the first agent, probably want an average of the two
            ep_reward += reward
            if self.eval_mode and any(reward < -1):
                # print("Negative reward agent " + str(np.argmin(reward)), " step ", step)
                self.last_run["greedy"] += 1

            if info['survival'] and info['donationBox_full'] and not job_done:
                job_done = True
                if self.args.tb_log: self.summary_w.add_scalar('Training/Job_Done',
                                                               (step - np.floor(
                                                                   step / self.args.max_steps) * self.args.max_steps),
                                                               self.metrics["global_step"])

            reward = _array_to_dict_tensor(self.agents, reward, self.device)
            done = _array_to_dict_tensor(self.agents, done, self.device)
            if not self.eval_mode:
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
                self.last_run["apple_history"] = np.array(apple_history)
                self.last_run["donation_box_history"] = np.array(donation_box_history)

                self.metrics["ep_count"] += 1
                self.metrics["reward_per_agent"].append(ep_reward)
                self.last_run["reward_per_agent"] = ep_reward
                self.metrics["reward_q"].append(np.mean(ep_reward))

                if self.eval_mode:
                    self.last_run["survive"] = info["survival"]
                    self.last_run["donation_box_full"] = info["donationBox_full"]
                    # Weak convergence
                    self.last_run["convergence"] = True if info["survival"] and not self.last_run["greedy"] \
                                                           and self.last_run["donation_box_full"] else False
                    return

                if not job_done:
                    if self.args.tb_log: self.summary_w.add_scalar('Training/Job_Done',
                                                                   (step - np.floor(
                                                                       step / self.args.max_steps) * self.args.max_steps),
                                                                   self.metrics["global_step"])
                job_done = False
                if step != self.args.n_steps - 1:
                    apple_history = []
                    donation_box_history = []
                else:
                    apple_history = np.array(apple_history).reshape(self.args.max_steps, self.args.n_agents, 1)
                    donation_box_history = np.array(donation_box_history).reshape(self.args.max_steps, 1)
                self.metrics["avg_reward"].append(np.mean(self.metrics["reward_q"]))
                record = {
                    'Training/Global_Step': self.metrics["global_step"],
                    'Training/Avg_Reward': np.mean(self.metrics["reward_q"]),
                }
                # add agent's reward to the record
                for k in self.agents:
                    record['Agent_' + str(k) + '/Reward'] = ep_reward[k]

                if self.args.tb_log:
                    # Add record to tensorboard
                    for k, v in record.items():
                        self.summary_w.add_scalar(k, v, self.metrics["global_step"])

                # if args.wandb_log: wandb.log(record)

                if self.args.verbose:
                    print(f"E: {self.metrics['ep_count']},\n\t "
                          f"Reward per agent: {ep_reward},\n\t "
                          f"Avg_Reward for all episodes: {record['Training/Avg_Reward']},\n\t "
                          f"Global_Step: {self.metrics['global_step']},\n\t "
                          )

                ep_reward = np.zeros(self.args.n_agents)
                # Reset environment
                observation = self.environment_reset()

    def _profile_subprocess(self, task):
        import cProfile
        profiler = cProfile.Profile()
        profiler.runcall(self._parallel_sim, task)
        profiler.dump_stats(f'profile_{task[-1]}')

    def _parallel_sim(self, tasks):
        env, result, env_id = tasks
        th.set_num_threads(1)
        data = {"global_step": self.metrics["global_step"], "logs": []}
        single_buffer = {k: Buffer(self.o_size, self.args.max_steps, self.args.max_steps, self.args.gamma,
                                   self.args.gae_lambda, self.device) for k in self.agents}
        start_time = time.time()
        data["global_step"] += env_id * self.args.max_steps
        if self.eval_mode:
            self.args.tb_log = False
        observation = self.environment_reset(env=env)

        action, logprob, s_value = [{k: 0 for k in self.agents} for _ in range(3)]
        env_action, ep_reward = [np.zeros(self.args.n_agents) for _ in range(2)]

        for step in range(self.args.max_steps):
            data["global_step"] += 1
            with th.no_grad():
                for k in self.agents:
                    (
                        env_action[k],
                        action[k],
                        logprob[k],
                        _,
                    ) = self.actor[k].get_action(observation[k])
                    if self.args.past_actions_memory > 0:
                        pass  # TODO: review this
                        # self.past_actions_memory[k].appendleft(float(action[k] / len(ACTIONS)))
                    if not self.eval_mode:
                        s_value[k] = self.critic[k](observation[k])

            non_tensor_observation, reward, done, info = env.step(env_action)
            if self.args.past_actions_memory > 0:
                for k in self.agents:
                    non_tensor_observation[k] = np.append(non_tensor_observation[k], self.past_actions_memory[k])

            ep_reward += reward

            reward = _array_to_dict_tensor(self.agents, reward, self.device)
            done = _array_to_dict_tensor(self.agents, done, self.device)
            if not self.eval_mode:
                for k in self.agents:
                    single_buffer[k].store(
                        observation[k],
                        action[k],
                        logprob[k],
                        reward[k],
                        s_value[k],
                        done[k]
                    )

            observation = _array_to_dict_tensor(self.agents, non_tensor_observation, self.device)

        # End of simulation
        data["reward_per_agent"] = ep_reward
        data["reward_q"] = np.mean(ep_reward)
        record = {
            'Training/Global_Step': data["global_step"],
            'Training/Avg_Reward': np.mean(data["reward_q"]),
        }
        # add agent's reward to the record
        for k in self.agents:
            record['Agent_' + str(k) + '/Reward'] = ep_reward[k]

        if self.args.tb_log:
            # Add record to tensorboard
            for k, v in record.items():
                data["logs"].append((k, v, data["global_step"]))
        data["single_buffer"] = single_buffer
        result[env_id] = data
        cname = mp.current_process().name
        if self.args.verbose:
            print(f"{cname} Time: {time.time() - start_time}")

    # TODO: Play from file


if __name__ == "__main__":
    args = config.parse_ppo_args()
    # Print all the arguments, so they are visible in the log
    print(args)

    # ppo = IPPO(args, env=gym.make(args.env))

    # Load environment parameters and create environment if needed
    env_params = config.parse_env_args()
    env = gym.make(args.env, **vars(env_params))

    ppo = IPPO(args, env=env)
    ppo.train()
