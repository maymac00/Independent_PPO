import abc
import time
from copy import deepcopy

import numpy as np
from PPO import PPOAgentExtraInfo, PPOAgent
from .IPPO import IPPO
import torch.multiprocessing as mp
import torch as th
from collections import deque

from .callbacks import UpdateCallback
from contextlib import contextmanager


class ParallelIPPO(IPPO, abc.ABC):
    def __init__(self, agents: list[PPOAgent] | list[PPOAgentExtraInfo], env, tot_steps, batch_size, **kwargs):
        super().__init__(agents, env, tot_steps, batch_size, **kwargs)
        self.n_cpus = mp.cpu_count()
        self.n_workers = min(self.n_agents, mp.cpu_count())
        print("Number of workers: ", self.n_workers)

        self.batch_size_for_worker = self.batch_size // self.n_workers
        # Change the size of each agent buffer
        for ag in self.agents:
            ag.buffer.resize(self.batch_size_for_worker)

    @contextmanager
    def set_torch_threads(self, num_threads):
        """Context manager to temporarily set the number of threads for PyTorch."""
        original_threads = th.get_num_threads()
        th.set_num_threads(num_threads)
        try:
            yield
        finally:
            th.set_num_threads(original_threads)

    def rollout(self):
        raise NotImplementedError("Parallel IPPO does not use this method.")

    def train(self):
        """
        Rollout the agents in the environment. Calls the _single_rollout method parallelized.
        Returns
        -------
        """
        # Use queues to send agents a dict of state_dict's to the workers every X updates
        manager = mp.Manager()
        model_queues = manager.Queue()
        buffer_queues = manager.Queue()

        processes = []
        for worker_id in range(self.n_workers):
            p = mp.Process(target=self.worker,
                           args=(worker_id, buffer_queues, model_queues))
            p.start()
            processes.append(p)

        # Get the buffer updates from the queue. Wait for all workers to send their buffers
        buffers = deque(maxlen=self.n_workers)

        while self.update_count < self.n_updates:
            try:
                while not buffer_queues.empty():
                    buffers.append(buffer_queues.get())

                if len(buffers) >= self.n_workers:
                    th.set_num_threads(self.n_cpus - self.n_workers)
                    for k, ag in enumerate(self.agents):
                        ag.buffer = sum([buffers[i][k] for i in range(1, self.n_workers)], buffers[0][k])
                    self.update_count += 1

                    buffers.clear()

                    for callback in self.callbacks:
                        if isinstance(callback, UpdateCallback):
                            callback.before_update()

                    obs = self.env.reset()[0]
                    th.set_num_threads(self.n_cpus-self.n_workers)
                    for i, agent in enumerate(self.agents):
                        self.update_agent(agent, obs[i])


                    for callback in self.callbacks:
                        if isinstance(callback, UpdateCallback):
                            callback.after_update()

                    # Send updated weights to workers
                    model = {"actors": [ag.actor.state_dict() for ag in self.agents],
                             "critics": [ag.critic.state_dict() for ag in self.agents]}
                    for i in range(self.n_workers):
                        model_queues.put(model)
                    th.set_num_threads(1)

            except KeyboardInterrupt:
                break
        # Stop workers
        for p in processes:
            p.terminate()
            


    def worker(self, worker_id, buffer_queues, model_queues):
        """
        Worker function for parallel rollout.
        Parameters
        """

        np.random.seed(worker_id)

        with self.set_torch_threads(1):

            waiting_model = False
            while True:
                try:
                    # get updated model weights
                    model = model_queues.get_nowait()
                    for i, ag in enumerate(self.agents):
                        ag.actor.load_state_dict(model["actors"][i])
                        ag.critic.load_state_dict(model["critics"][i])
                    waiting_model = False
                except Exception as e:
                    if waiting_model:
                        continue

                # Clear the buffers
                for ag in self.agents:
                    ag.buffer.clear()


                self._single_rollout(self.agents, self.env)
                # add dict of buffers to queue
                buffer_queues.put({i: ag.buffer for i, ag in enumerate(self.agents)})
                waiting_model = True
                pass


    @abc.abstractmethod
    def _single_rollout(self, agents, env):
        """
        Rollout the agents in the environment. Store transitions in the buffer.
        Returns
        -------
        """
        raise NotImplementedError