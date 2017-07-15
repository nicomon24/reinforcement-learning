'''
    This is the agent class, will extend normal player
'''

from player import *
import pickle

class ValueFuntion():

    def __init__(self):
        self.values = np.full(3**9, 0.5)

    def set_state(self, state, value):
        self.values[state] = value

    def get_state(self, state):
        return self.values[state]

class Agent(Player):

    def __init__(self, pid):
        Player.__init__(self, pid)
        self.vf = ValueFuntion()
        self.epsilon = 0.05
        self.current_history = []
        self.learning_rate = 0.05

    def play(self, env, verbose = True):
        if verbose:
            print("Player ", self.pid)
            print("Possible actions: ", env.action_space())
        # Add this state to history
        self.current_history.append(env.get_state())
        # Possible actions
        actions = env.action_space()
        # All the possible next states
        next_states = [ env.next_state(self.pid, i) for i in actions]
        # E-greedy step
        pivot = np.random.random()
        if pivot < self.epsilon:
            # Take random action
            j = np.random.choice(len(actions))
        else:
            # Take best action
            j = np.argmax([self.vf.get_state(s) for s in next_states])
        action = actions[j]
        self.current_history.append(env.next_state(self.pid, action))
        return env.take_action(self.pid, action)

    def assign_reward(self, reward):
        # Set the final node as the reward
        self.vf.set_state(self.current_history[-1], reward)
        nxt = reward
        # Backward propagation of reward
        for state in self.current_history[-2::-1]:
            value = self.vf.get_state(state)
            self.vf.set_state(state, value + self.learning_rate * (nxt - value))
            nxt = value
        self.current_history = []
