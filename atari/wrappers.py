'''
    OpenAI gym environment wrappers for:
    - Skip N frames
    - Crop to 83x83
    - Fire on reset
    - Stack N frames in time
'''

import gym
from collections import deque
import numpy as np
from scipy.misc import imresize

class NoopEnvironment(gym.Wrapper):

    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max

    def _reset(self):
        self.env.reset()
        noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class SkipperEnvironment(gym.Wrapper):

    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self.skip = skip
        # Max the last 2 frames to avoid flickering
        self.observation_buffer = deque(maxlen=2)

    def _step(self, action):
        skip_reward = 0.0
        done = False
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            self.observation_buffer.append(obs)
            skip_reward += reward
            if done:
                break
        # Now take the max of the 2 last frames
        max_observation = np.max(np.stack(self.observation_buffer), axis=0)
        return max_observation, skip_reward, done, info

    def _reset(self):
        self.observation_buffer.clear()
        obs = self.env.reset()
        self.observation_buffer.append(obs)
        return obs

class CroppedEnvironment(gym.ObservationWrapper):

    def __init__(self, env, target_size):
        gym.ObservationWrapper.__init__(self, env)
        self.target_size = target_size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.target_size, self.target_size, 1))

    def _observation(self, obs):
        bw = obs.mean(axis=2, dtype=np.uint8)
        cropped = bw[30:195,4:-4]
        resized = imresize(cropped, size=(self.target_size, self.target_size), interp='nearest')
        return np.reshape(resized, [self.target_size, self.target_size, 1])

class TimeStackEnvironment(gym.Wrapper):

    def __init__(self, env, time_period=4):
        gym.Wrapper.__init__(self, env)
        self.time_period = time_period
        self.time_stack = deque([], maxlen=time_period)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.env.observation_space.shape[0], self.env.observation_space.shape[0], time_period))

    def _reset(self):
        obs = self.env.reset()
        for i in range(self.time_period):
            self.time_stack.append(obs)
        return np.concatenate(self.time_stack, axis=2)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.time_stack.append(obs)
        return np.concatenate(self.time_stack, axis=2), reward, done, info

class FiringEnvironment(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def _reset(self):
        # Fire an action
        self.env.reset()
        obs, reward, done, info = self.env.step(1)
        if done:
            obs = self.env.reset()
        return obs

# Episode is a single life
class NoCatsLivesEnvironment(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.real_done = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.real_done = done
        # Check current lives
        lives = self.env.unwrapped.ale.lives()
        # Check if we lost a life in the last action
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        if self.real_done:
            obs = self.env.reset()
        else:
            # Simulate a reset with a noop
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)
