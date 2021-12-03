import numpy as np


class Trajectory(object):
    """
    将[(obs,act,rew,done,info,next_obs),...]重构，方便sample
    """

    def __init__(self, trajectory_data) -> None:
        super().__init__()
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.next_observations = []
        for data in trajectory_data:
            self.observations.append(data[0])
            self.actions.append(data[1])
            self.rewards.append(data[2])
            self.dones.append(data[3])
            self.infos.append(data[4])
            self.next_observations.append(data[5])

    def __getitem__(self, key):
        return {
            "obs": self.observations[key],
            "act": self.actions[key],
            "rew": self.rewards[key],
            "done": self.dones[key],
            "info": self.infos[key],
            "next_obs": self.next_observations[key],
        }

    def __len__(self):
        return len(self.observations)


class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Args:
            capacity:
        """
        self.capacity = capacity
        self.trajectories = []

    def add(self, trajectory):
        """
        Add trajectory to env 
        """
        self.trajectories.append(Trajectory(trajectory))
        print(trajectory)

    def sample(self, batch_size):
        """
        Sample from buffer
        """
        trajectory_indices = np.random.choice(
            len(self.trajectories), batch_size,
        )
        batch = {
            "obs": [],
            "act": [],
            "rew": [],
            "done": [],
            "info": [],
            "next_obs": [],
        }
        for index in trajectory_indices:
            data_index = np.random.randint(0, len(self.trajectories[index]))
            data = self.trajectories[index][data_index]
            for key, value in data.items():
                batch[key].append(value)
        return batch

