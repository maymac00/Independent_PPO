import argparse
import copy
import json
import logging
import os
import time
import warnings

from collections import deque
from typing import List, Type, Dict

import numpy as np

from .agent import SoftmaxActor, Critic, Agent

from .utils.memory import Buffer
from .utils.misc import *
from .utils.memory import Buffer
from .utils.misc import Array, set_torch, normalize, set_seeds
import torch.nn as nn
import IndependentPPO.config as config
from .callbacks import UpdateCallback, Callback
import torch as th
import IndependentPPO.config as config
from .callbacks import UpdateCallback, Callback

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


class IPPO:
    callbacks: List[Callback] = []

    @staticmethod
    def actors_from_file(folder, dev='cpu', eval=True):
        """
        Creates the actors from the folder's model, and returns them set to eval mode.
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
                actor = SoftmaxActor(o_size, a_size, args.h_size, args.h_layers, eval=eval).to(dev)
                actor.load_state_dict(model)

                agents.append(actor)
            return agents

    @staticmethod
    def agents_from_file(folder, dev='cpu', eval=True):
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
                actor = SoftmaxActor(o_size, a_size, args.h_size, args.h_layers,  eval=eval).to(dev)
                actor.load_state_dict(model)
                critic = Critic(o_size, args.h_size, args.h_layers).to(dev)
                critic.load_state_dict(th.load(folder + f"/critic_{k}.pth"))

                agents.append(Agent(actor, critic, args.actor_lr, args.critic_lr))
            return agents

    def __init__(self, args, env, run_name=None):

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.ERROR)
        # Check if the logger already has a handler
        if len(self.logger.handlers) == 0:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        if type(args) is dict:
            args = argparse.Namespace(**args)
        elif type(args) is argparse.Namespace:
            args = args
        elif type(args) is str:
            args = config.args_from_json(args)
        self.init_args = args

        for k, v in self.init_args.__dict__.items():
            setattr(self, k, v)
        self.entropy_value = self.ent_coef
        if run_name is not None:
            self.run_name = run_name
        else:
            # Get and format day and time
            timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
            self.run_name = f"{self.env_name}__{self.tag}__{self.seed}__{timestamp}__{np.random.randint(0, 100)}"

        # Action-Space
        self.o_size = env.observation_space.sample().shape[0]
        self.a_size = env.action_space.n

        # Attributes
        self.r_agents = range(self.n_agents)
        self.run_metrics = None
        self.update_metrics = {}
        self.sim_metrics = {}
        self.folder = None
        self.eval_mode = False

        # Behaviours
        self.lr_scheduler = None
        self.private_callbacks = []

        #   Torch init
        self.device = set_torch(self.n_cpus, self.cuda)

        #   Actor-Critic
        self.n_updates = None
        self.buffer = None
        self.agents, self.buffer = {}, {}

        for k in self.r_agents:
            self.agents[k] = Agent(
                SoftmaxActor(self.o_size, self.a_size, self.h_size, self.h_layers).to(self.device),
                Critic(self.o_size, self.h_size, self.h_layers).to(self.device),
                self.actor_lr,
                self.critic_lr,
            )
            self.buffer[k] = Buffer(self.o_size, self.n_steps, self.max_steps, self.gamma,
                                    self.gae_lambda, self.device)

        self.env = env

    def environment_reset(self, env=None):
        if env is None:
            non_tensor_observation, info = self.env.reset()
            observation = _array_to_dict_tensor(self.r_agents, non_tensor_observation, self.device)
            return observation
        else:
            non_tensor_observation, info = env.reset()
            observation = _array_to_dict_tensor(self.r_agents, non_tensor_observation, self.device)
            return observation

    def update(self):
        # Run callbacks
        for c in IPPO.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.before_update()

        for c in self.private_callbacks:
            if issubclass(type(c), UpdateCallback):
                c.before_update()

        th.set_num_threads(self.n_cpus)
        update_metrics = {}

        with th.no_grad():
            for k in self.r_agents:
                if self.agents[k].isFrozen():
                    continue
                value_ = self.agents[k].critic(self.environment_reset()[k])
                self.buffer[k].compute_mc(value_.reshape(-1))

        # Optimize the policy and value networks
        for k in self.r_agents:
            self.buffer[k].clear()
            if self.agents[k].isFrozen():
                continue
            b = self.buffer[k].sample()

            # Actor optimization
            for epoch in range(self.n_epochs):
                _, _, logprob, entropy = self.agents[k].actor.get_action(b['observations'], b['actions'])
                entropy_loss = entropy.mean()

                update_metrics[f"Agent_{k}/Entropy"] = entropy_loss.detach()

                logratio = logprob - b['logprobs']
                ratio = logratio.exp()
                update_metrics[f"Agent_{k}/Ratio"] = ratio.mean().detach()

                mb_advantages = b['advantages']
                if self.norm_adv: mb_advantages = normalize(mb_advantages)

                actor_loss = mb_advantages * ratio
                update_metrics[f"Agent_{k}/Actor Loss Non-Clipped"] = actor_loss.mean().detach()

                clipped_ratios = th.clamp(ratio, 1 - self.clip, 1 + self.clip)
                actor_clip_loss = mb_advantages * clipped_ratios

                # Log percent of clipped ratio
                update_metrics[f"Agent_{k}/Clipped Ratio"] = ((ratio < (1 - self.clip)).sum().item() + (ratio > (1 + self.clip)).sum().item()) / np.prod(ratio.shape)

                # Calculate clip fraction
                actor_loss = th.min(actor_loss, actor_clip_loss).mean()
                update_metrics[f"Agent_{k}/Actor Loss"] = actor_loss.detach()

                actor_loss = -actor_loss - self.entropy_value * entropy_loss
                update_metrics[f"Agent_{k}/Actor Loss with Entropy"] = actor_loss.detach()

                self.agents[k].a_optimizer.zero_grad(True)
                actor_loss.backward()

                if self.log_gradients:
                    for name, param in self.agents[k].actor.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            update_metrics[f"Agent_{k}/ZGradients: {name}"] = grad_norm

                nn.utils.clip_grad_norm_(self.agents[k].actor.parameters(), self.max_grad_norm)
                self.agents[k].a_optimizer.step()

            # Critic optimization
            for epoch in range(self.n_epochs * self.critic_times):
                values = self.agents[k].critic(b['observations']).squeeze()

                # Value clipping
                if self.clip_vloss:
                    v_loss_unclipped = (values - b['returns']) ** 2

                    v_clipped = (th.clamp(values, b['values'] - self.clip, b['values'] + self.clip) - b['returns']) ** 2
                    v_loss_clipped = th.min(v_loss_unclipped, v_clipped)

                    # Log percent of clipped ratio
                    update_metrics[f"Agent_{k}/Critic Clipped Ratio"] = ((values < (b['values'] - self.clip)).sum().item() + (
                                values > (b['values'] + self.clip)).sum().item()) / np.prod(values.shape)

                    critic_loss = 0.5 * v_loss_clipped.mean()
                    update_metrics[f"Agent_{k}/Critic Loss Non-Clipped"] = critic_loss.detach()
                else:
                    # No value clipping
                    critic_loss = 0.5 * ((values - b['returns']) ** 2).mean()

                update_metrics[f"Agent_{k}/Critic Loss"] = critic_loss.detach()

                critic_loss = critic_loss * self.v_coef

                self.agents[k].c_optimizer.zero_grad(True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.agents[k].critic.parameters(), self.max_grad_norm)
                self.agents[k].c_optimizer.step()

            loss = actor_loss - entropy_loss * self.entropy_value + critic_loss
            update_metrics[f"Agent_{k}/Loss"] = loss.detach().cpu()
        self.update_metrics = update_metrics
        mean_loss = np.array([self.update_metrics[f"Agent_{k}/Loss"] if not self.agents[k].isFrozen() else 0 for k in
                              self.r_agents]).mean()
        self.run_metrics["mean_loss"].append(mean_loss)

        # Run callbacks
        for c in IPPO.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.after_update()

        for c in self.private_callbacks:
            if issubclass(type(c), UpdateCallback):
                c.after_update()

        return update_metrics

    def rollout(self):
        sim_metrics = {"reward_per_agent": np.zeros(self.n_agents)}

        observation = self.environment_reset()

        action, logprob, s_value = [{k: 0 for k in self.r_agents} for _ in range(3)]
        env_action = np.zeros(self.n_agents)
        ep_reward = np.zeros(self.n_agents)

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
                sim_metrics["reward_per_agent"] += ep_reward
                ep_reward = np.zeros(self.n_agents)
                # Reset environment
                observation = self.environment_reset()
        sim_metrics["reward_per_agent"] /= (self.n_steps / self.max_steps)

        self.run_metrics["avg_reward"].append(sim_metrics["reward_per_agent"].mean())
        # Save mean reward per agent
        for k in self.r_agents:
            self.run_metrics["agent_performance"][f"Agent_{k}/Reward"] = sim_metrics["reward_per_agent"][k].mean()
        return np.array(
            [self.run_metrics["agent_performance"][f"Agent_{self.r_agents[k]}/Reward"] for k in self.r_agents])

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
                    Critic(self.o_size, self.h_size, self.h_layers).to(self.device),
                    self.init_args.actor_lr,
                    self.init_args.critic_lr,
                )
                self.buffer[k] = Buffer(self.o_size, self.n_steps, self.max_steps, self.gamma,
                                        self.gae_lambda, self.device)
        else:
            self.agents = set_agents

        # Reset run metrics:
        self.run_metrics = {
            'global_step': 0,
            'ep_count': 0,
            'start_time': time.time(),
            'avg_reward': deque(maxlen=500),
            'agent_performance': {},
            'mean_loss': deque(maxlen=500),
        }

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
        self.reward_shape = (self.env.n_agents, self.env.reward_space.shape[0])
        # TODO: Set the action space, translate actions to env_actions

    def _finish_training(self):
        # Log relevant data from training
        self.logger.info(f"Training finished in {time.time() - self.run_metrics['start_time']} seconds")
        self.logger.info(f"Average reward: {np.mean(self.run_metrics['avg_reward'])}")
        self.logger.info(f"Average loss: {np.mean(self.run_metrics['mean_loss'])}")
        self.logger.info(f"Std mean loss: {np.std(self.run_metrics['mean_loss'])}")
        self.logger.info(f"Number of episodes: {self.run_metrics['ep_count']}")
        self.logger.info(f"Number of updates: {self.n_updates}")

        self.save_experiment_data()

    def save_experiment_data(self, folder=None, ckpt=False):
        config = self.init_args
        # Create new folder in to save the model using tag, n_steps, tot_steps and seed as name
        if folder is None:
            folder = f"{config.save_dir}/{config.tag}/{config.n_steps}_{config.tot_steps // config.max_steps}_{config.seed}"

        # Check if folder's config file is the same as the current config
        def diff_config(path):
            if os.path.exists(path):
                tries = 0
                while tries < 5:
                    try:
                        with open(path + "/config.json", "r") as f:
                            old_config = json.load(f)
                        if old_config != vars(config):
                            return True
                        return False
                    except FileNotFoundError as e:
                        tries += 1
                        time.sleep(1)
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
            # Try-catch for concurrency issues
            tries = 0
            while tries < 5:
                try:
                    os.makedirs(folder)
                    break
                except FileExistsError as e:
                    tries += 1
                    time.sleep(1)

        print(f"Saving model in {folder}")
        self.folder = folder
        setattr(config, "saved_dir", folder)
        setattr(self, "saved_dir", folder)

        # Save the model
        for k in self.r_agents:
            th.save(self.agents[k].actor.state_dict(), folder + f"/actor_{k}.pth")
            th.save(self.agents[k].critic.state_dict(), folder + f"/critic_{k}.pth")
            self.agents[k].save_dir = folder

        # Save the args as a json file
        with open(folder + "/config.json", "w") as f:
            json.dump(vars(config), f, indent=4)
        return folder

    def addCallbacks(self, callbacks, private=False):
        if private:
            if isinstance(callbacks, list):
                for c in callbacks:
                    if not issubclass(type(c), Callback):
                        raise TypeError("Element of class ", type(c).__name__, " not a subclass from Callback")
                    c.ppo = self
                    c.initiate()
                self.private_callbacks = callbacks
            elif isinstance(callbacks, Callback):
                callbacks.ppo = self
                callbacks.initiate()
                self.private_callbacks.append(callbacks)
            else:
                raise TypeError("Callbacks must be a Callback subclass or a list of Callback subclasses")
        else:
                if isinstance(callbacks, list):
                    for c in callbacks:
                        if not issubclass(type(c), Callback):
                            raise TypeError("Element of class ", type(c).__name__, " not a subclass from Callback")
                        c.ppo = self
                        c.initiate()
                    IPPO.callbacks = callbacks
                elif isinstance(callbacks, Callback):
                    callbacks.ppo = self
                    callbacks.initiate()
                    IPPO.callbacks.append(callbacks)
                else:
                    raise TypeError("Callbacks must be a Callback subclass or a list of Callback subclasses")

    def load_checkpoint(self, folder):
        # Load the args from the folder
        with open(folder + "/config.json", "r") as f:
            args = argparse.Namespace(**json.load(f))
            # Load the model
            agents = {}
            for k in range(args.n_agents):
                model_actor = th.load(folder + f"/actor_{k}.pth")
                o_size = model_actor["hidden.0.weight"].shape[1]
                a_size = model_actor["output.weight"].shape[0]
                actor = SoftmaxActor(o_size, a_size, args.h_size, args.h_layers, eval=True).to(self.device)
                actor.load_state_dict(model_actor)

                model_critic = th.load(folder + f"/critic_{k}.pth")
                critic = Critic(o_size, args.h_size, args.h_layers).to(self.device)
                critic.load_state_dict(model_critic)

                agents[k] = Agent(actor, critic, args.actor_lr, args.critic_lr)
            self.agents = agents
