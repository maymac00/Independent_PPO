import time
import logging

import numpy as np

from .CallbackI import UpdateCallback

class PrintInfo(UpdateCallback):

    def __init__(self, ippo, freq=1):
        super().__init__(ippo)
        self.freq = freq
        self.t0 = time.time()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)


    def after_update(self):
        pass

    def before_update(self):
        t1 = time.time()
        if self.ippo.update_count % self.freq == 0 and self.ippo.update_count > 0:
            rewards = np.array([0.] * self.ippo.n_agents)

            for i, ag in enumerate(self.ippo.agents):
                rewards[i] = ag.buffer.buffer_episodic_return()
            sps = self.ippo.batch_size*self.freq / (t1 - self.t0)
            sps = round(sps, 2)
            print(f"Update: {self.ippo.update_count}, SPS: {sps}, Rewards: {rewards.round(2)}")
        self.t0 = t1
