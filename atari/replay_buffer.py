'''
    Implementation of the Replay Buffer
'''

import random
from collections import deque

class ReplayBuffer:

    def __init__(self, max_size):
        self.buffer = deque([], maxlen = max_size)

    def add_sample(self, sample):
        # Add the new sample
        self.buffer.append(sample)

    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
