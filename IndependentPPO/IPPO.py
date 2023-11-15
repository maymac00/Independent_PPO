import argparse
import copy
import json
import time
import warnings

from collections import deque

from torch import optim

from IndependentPPO.agent import SoftmaxActor, Critic

from IndependentPPO.utils.memory import Buffer, merge_buffers
from IndependentPPO.utils.misc import *
import torch.nn as nn
from torch.multiprocessing import Manager
import torch.multiprocessing as mp
import IndependentPPO
from IndependentPPO.wrappers import UpdateCallback, Callback

# The MA environment does not follow the gym SA scheme, so it raises lots of warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def _array_to_dict_tensor(agents: List[int], data: Array, device: th.device, astype: Type = th.float32) -> Dict:
    # Check if the provided device is already the current device
    is_same_device = (device == th.cuda.current_device()) if device.type == 'cuda' else (device == th.device('cpu'))

    if is_same_device:
        return {k: th.as_tensor(d, dtype=astype) for k, d in zip(agents, data)}
    else:
        return {k: th.as_tensor(d, dtype=astype).to(device) for k, d in zip(agents, data)}


import torch


def find_tensors_requiring_grads(obj, parent_name=''):
    """
    Recursively find all tensors with requires_grad=True in a PyTorch module or object.
    """
    tensors_requiring_grad = []

    # If the object itself is a tensor that requires grad, add it to the list
    if isinstance(obj, torch.Tensor) and obj.requires_grad:
        return [(parent_name, obj)]

    # If this is a module, we'll look at its parameters and buffers
    if isinstance(obj, torch.nn.Module):
        for name, param in obj.named_parameters(recurse=False):
            if param.requires_grad:
                tensors_requiring_grad.append((f'{parent_name}.{name}' if parent_name else name, param))
        for name, buffer in obj.named_buffers(recurse=False):
            if buffer.requires_grad:
                tensors_requiring_grad.append((f'{parent_name}.{name}' if parent_name else name, buffer))

    # Recursively check any iterable or object attributes
    if hasattr(obj, '__dict__'):
        for name, attr in obj.__dict__.items():
            tensors_requiring_grad += find_tensors_requiring_grads(attr,
                                                                   f'{parent_name}.{name}' if parent_name else name)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, torch.Tensor)):
        for idx, item in enumerate(obj):
            tensors_requiring_grad += find_tensors_requiring_grads(item, f'{parent_name}[{idx}]')

    return tensors_requiring_grad


class IPPO:

    @staticmethod
    def agents_from_file(folder, dev='cpu'):
        """
        Creates the agents from the folder's model, and returns them set to eval mode.
        It is assumed that the model is a SoftmaxActor from file agent.py which only has hidden layers and an output layer.
        :return:
        """
        # Load the args from the folder
        with open(folder + "/config.json", "r") as f:
            args = argparse.Namespace(**json.load(f))
            # Load the model
            agents = []
            for k in range(args.n_agents):
                model = th.load(folder + f"/actor_{k}.pth")
                o_size = model["hidden.0.weight"].shape[1]
                a_size = model["output.weight"].shape[0]
                actor = SoftmaxActor(o_size, a_size, args.h_size, args.h_layers, eval=True).to(dev)
                actor.load_state_dict(model)
                agents.append(actor)
            return agents

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
        self.entropy_value = self.ent_coef
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
        self.update_metrics = {}
        self.sim_metrics = {}
        self.callbacks = []

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

        # Run callbacks
        for c in self.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.before_update()

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
        self.update_metrics = update_metrics

        # Run callbacks
        for c in self.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.after_update()

        return update_metrics

    def _sim(self):
        sim_metrics = {"reward_per_agent": np.zeros(self.n_agents)}
        episodes = 0
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
                episodes += 1
                sim_metrics["reward_per_agent"] += ep_reward
                ep_reward = np.zeros(self.n_agents)
                # Reset environment
                observation = self.environment_reset()
        sim_metrics["reward_per_agent"] /= episodes
        return sim_metrics

    def _parallel_sim(self, tasks):
        env, result, env_id = tasks
        th.set_num_threads(1)
        data = {"global_step": self.run_metrics["global_step"], "reward_per_agent": None}

        single_buffer = {k: Buffer(self.o_size, self.max_steps, self.max_steps, self.gamma,
                                   self.gae_lambda, self.device) for k in self.agents}

        data["global_step"] += env_id * self.max_steps

        observation = self.environment_reset(env=env)

        action, logprob, s_value = [{k: 0 for k in self.agents} for _ in range(3)]
        env_action, ep_reward = [np.zeros(self.n_agents) for _ in range(2)]

        for step in range(self.max_steps):
            data["global_step"] += 1

            with th.no_grad():
                for k in self.agents:
                    (
                        env_action[k],
                        action[k],
                        logprob[k],
                        _,
                    ) = self.actor[k].get_action(observation[k])

                    s_value[k] = self.critic[k](observation[k])

            non_tensor_observation, reward, done, info = env.step(env_action)

            ep_reward += reward

            reward = _array_to_dict_tensor(self.agents, reward, self.device)
            done = _array_to_dict_tensor(self.agents, done, self.device)
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
        for k in self.agents:
            single_buffer[k].detach()
        data["reward_per_agent"] = ep_reward
        data["single_buffer"] = single_buffer
        result[env_id] = data
        pass

    def train(self):
        self.environment_setup()
        # set seed for training
        set_seeds(self.seed, self.th_deterministic)

        # Init actor-critic setup
        self.actor, self.critic, self.a_optim, self.c_optim, self.buffer = {}, {}, {}, {}, {}

        for k in self.agents:
            self.actor[k] = SoftmaxActor(self.o_size, self.a_size, self.h_size, self.h_layers).to(
                self.device)
            self.a_optim[k] = optim.Adam(list(self.actor[k].parameters()), lr=self.actor_lr, eps=1e-5)
            self.critic[k] = Critic(self.o_size, self.h_size, self.h_layers).to(self.device)
            self.c_optim[k] = optim.Adam(list(self.critic[k].parameters()), lr=self.critic_lr, eps=1e-5)
            self.buffer[k] = Buffer(self.o_size, self.n_steps, self.max_steps, self.gamma,
                                    self.gae_lambda, self.device)

        # Reset run metrics:
        self.run_metrics = {
            'global_step': 0,
            'ep_count': 0,
            'start_time': time.time(),
            'sim_start_time': time.time(),
            'avg_reward': deque(maxlen=500),
        }

        # Training loop
        self.n_updates = self.tot_steps // self.batch_size
        for update in range(1, self.n_updates + 1):
            self.run_metrics['sim_start_time'] = time.time()
            if not self.parallelize:
                self.sim_metrics = self._sim()
                self.run_metrics['avg_reward'].append(self.sim_metrics["reward_per_agent"].mean())
            else:

                with Manager() as manager:
                    d = manager.dict()
                    batch_size = int(self.n_steps / self.max_steps)

                    tasks = [(self.env, d, i) for i in range(batch_size)]
                    solved = 0
                    while solved < batch_size:
                        runs = min(self.n_envs, batch_size - solved)
                        # print("Running ", runs, " tasks")
                        with mp.Pool(runs) as p:
                            p.map(self._parallel_sim, tasks[:runs])
                        # Remove solved tasks
                        solved += runs
                        tasks = tasks[runs:]

                    # Fetch the logs
                    self.sim_metrics = self._parallel_results(d, batch_size)

                    # Merge the results
                    for k in self.agents:
                        self.buffer[k] = merge_buffers([d[i]["single_buffer"][k] for i in range(batch_size)])
                self.run_metrics['ep_count'] += solved
                self.run_metrics['global_step'] += solved * self.max_steps
                self.run_metrics['avg_reward'].append(
                    np.array([np.mean(s["reward_per_agent"]) for s in self.sim_metrics]).mean())
            self.update()

        self.finish_training()

    def environment_setup(self):
        if self.env is None:
            raise Exception("Environment not set")
        obs, info = self.env.reset()
        # Decentralized learning checks
        if isinstance(obs, list):
            if len(obs) != self.n_agents:
                raise Exception("The environment returns a list of observations but the number of agents "
                                "is not the same as the number of observations.")
        elif isinstance(obs, np.ndarray):
            if len(obs.shape) != 2:
                raise Exception("The environment returns a numpy array of observations but the shape is not 2D. It "
                                "should be (agents x observation).")
        else:
            raise Exception("Observation is not a list neither an array.")

        self.o_size = self.env.observation_space.sample().shape[0]
        self.a_size = self.env.action_space.n
        print(f"Observation space: {self.o_size}, Action space: {self.a_size}")
        # TODO: Set the action space, translate actions to env_actions

    def _parallel_results(self, d, batch_size):
        sim_metrics = [{}] * batch_size
        for i in range(batch_size):
            sim_metrics[i]["reward_per_agent"] = d[i]["reward_per_agent"]
            sim_metrics[i]["global_step"] = d[i]["global_step"]
        return sim_metrics

    def finish_training(self):
        self.save_experiment_data()

    def save_experiment_data(self, folder=None, ckpt=False):
        config = self.args
        # Create new folder in to save the model using tag, n_steps, tot_steps and seed as name
        if folder is None:
            folder = f"{config.save_dir}/{config.tag}/{config.n_steps}_{config.tot_steps // config.max_steps}_{config.seed}"

        # Check if folder's config file is the same as the current config
        def diff_config(path):
            if os.path.exists(path):
                with open(path + "/config.json", "r") as f:
                    old_config = json.load(f)
                if old_config != vars(config):
                    return True
                return False
            return False

        num = 1
        if not ckpt:
            _folder = copy.copy(folder)
            while diff_config(_folder):
                # append a number to the folder name
                _folder = folder + "_(" + str(num) + ")"
                num += 1
            folder = _folder
        else:
            folder = folder + "_ckpt"

        if not os.path.exists(folder):
            os.makedirs(folder)

        print(f"Saving model in {folder}")

        # Save the model
        for k in range(config.n_agents):
            th.save(self.actor[k].state_dict(), folder + f"/actor_{k}.pth")
            th.save(self.critic[k].state_dict(), folder + f"/critic_{k}.pth")

        # Save the args as a json file
        with open(folder + "/config.json", "w") as f:
            json.dump(vars(config), f, indent=4)
        return folder

    def addCallbacks(self, callbacks):
        if isinstance(callbacks, list):
            for c in callbacks:
                if not issubclass(type(c), Callback):
                    raise TypeError("Element of class ", type(c).__name__, " not a subclass from Callback")
                c.ppo = self
                c.initiate()
            self.callbacks = callbacks
        elif isinstance(callbacks, Callback):
            callbacks.ppo = self
            callbacks.initiate()
            self.callbacks.append(callbacks)
        else:
            raise TypeError("Callbacks must be a Callback subclass or a list of Callback subclasses")
