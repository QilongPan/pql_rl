class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Args:
            capacity:
        """
        self.capacity = capacity
        self.trajectories = []
        pass

    def store_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def sample(self):
        pass
