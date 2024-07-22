import numpy as np
class ReplayBuffer():
    def __init__(self):
        self.replay_buffer = []
    
    def add(self, tup):
        self.replay_buffer.extend(tup)

    def sample(self, amount = 10):
        index = np.random.random_integers(0, len(self.replay_buffer)-amount)
        return np.array(self.replay_buffer)[index: index+amount]
    def save(self):
        arr = np.array(self.replay_buffer)
        np.save("ReplayBuffer", arr)
        