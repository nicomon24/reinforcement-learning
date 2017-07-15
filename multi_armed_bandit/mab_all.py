import numpy as np
import math
import matplotlib.pyplot as plt

class Bandit:

    def __init__(self, true_mean, optimistic=False):
        self.true_mean = true_mean
        self.mean = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.true_mean

    def update(self, x):
        self.N += 1
        self.mean = (1-1/self.N) * self.mean + 1/self.N * x

class BayesianBandit(Bandit):

    def __init__(self, true_mean):
        Bandit.__init__(self, true_mean)

class OptimisticBandit(Bandit):

    def __init__(self, true_mean):
        Bandit.__init__(self, true_mean)
        self.mean = 10

def run(strategy):

    m1 = 0.1
    m2 = 0.2
    m3 = 0.3
    N = 100000
    epsilon = 0.05

    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    if strategy == 'optimistic':
        bandits = [OptimisticBandit(m1), OptimisticBandit(m2), OptimisticBandit(m3)]
    elif strategy == 'bayesian':
        bandits = [BayesianBandit(m1), BayesianBandit(m2), BayesianBandit(m3)]

    data = np.empty(N)

    # Start epsilon-greedy loops
    for i in range(N):
        print([b.mean for b in bandits])
        if strategy=='greedy':
            pivot = np.random.random()
            if pivot < epsilon:
                # Take random action
                j = np.random.choice(3)
            else:
                # Take best action
                j = np.argmax([b.mean for b in bandits])
        else:
            j = np.argmax([b.mean for b in bandits])
        # Perform action, update mean
        x = bandits[j].pull()
        bandits[j].update(x)
        # Add data for plot
        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    for b in bandits:
        print(b.mean)

    return cumulative_average

egreedy = run('greedy')
gianni = run('optimistic')

# log scale plot
plt.plot(egreedy, label='e-greedy')
plt.plot(gianni, label='optimistic')
plt.legend()
plt.xscale('log')
plt.show()
