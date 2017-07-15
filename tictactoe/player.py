'''
    This is the base class of a Player, agent will extend this
    In this case, the action is taken by the user with a prompt
'''

from environment import *

import sys

class Player():

    def __init__(self, pid):
        self.pid = pid

    def play(self, env):
        print("Player ", self.pid)
        print("Possible actions: ", env.action_space())
        action = int(input("Choose an action:"))
        return env.take_action(self.pid, action)

    def assign_reward(self, reward):
        self.reward = reward
