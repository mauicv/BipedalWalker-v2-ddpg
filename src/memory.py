"""ReplayBuffer class

Uses circular memory storage to store states, actions, rewards and dones.
Implements sample that returns a batch of training samples for learning.
"""

import numpy as np
import random


class ReplayBuffer:
    def __init__(
            self,
            action_space_dim,
            state_space_dim,
            size=10000,
            sample_size=64):
        self.states = np.zeros((size, state_space_dim))
        self.next_states = np.zeros((size, state_space_dim))
        self.actions = np.zeros((size, action_space_dim))
        self.rewards = np.zeros(size)
        self.dones = np.zeros(size)
        self._end_index = 0
        self.sample_size = sample_size

    def push(self, state, next_state, action, reward, done):
        self.states[self._end_index % len(self.states)] = state
        self.next_states[self._end_index % len(self.states)] = next_state
        self.actions[self._end_index % len(self.states)] = action
        self.rewards[self._end_index % len(self.states)] = reward
        self.dones[self._end_index % len(self.states)] = done
        self._end_index += 1

    @property
    def ready(self):
        return self._end_index >= self.sample_size

    @property
    def full(self):
        return self._end_index >= len(self.states)

    def sample(self):
        length = self._end_index if not self.full else len(self.states)
        sample_size = self.sample_size if self.ready else self._end_index + 1
        random_integers = random.sample(range(length), sample_size)
        index_array = np.array(random_integers)
        state_samples = self.states[index_array]
        next_state_samples = self.states[index_array]
        action_samples = self.actions[index_array]
        reward_samples = self.rewards[index_array]
        done_samples = self.dones[index_array]
        return state_samples, next_state_samples, action_samples, \
            reward_samples, done_samples
