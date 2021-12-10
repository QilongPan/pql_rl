import time

from pql_rl.infer.base import InferServer

"""
use zmq
https://zeromq.org/socket-api/
"""


class BatchInfer(InferServer):
    """
    Collect a batch of data at regular intervals to make predictions
    """

    def __init__(self, policy, wait_time=0) -> None:
        super().__init__(policy)
        self.wait_time = wait_time
        self.batch_obs = []

    def get_actions(self, obs_ls):
        self.first_data_time = time.time()
        self.batch_obs.extend(obs_ls)
        if time.time() - self.first_data_time >= self.wait_time:
            self.first_data_time = time.time()
            self.policy(self.batch_obs)
            self.batch_obs.clear()
