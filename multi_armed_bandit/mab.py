import numpy as np
import matplotlib.pyplot as plt

class Bandit:

    def  __init__(self, true_mean):
        self.true_mean = true_mean
        self.pred_mean = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.true_mean

    def update(self, x):
        self.N += 1
        self.pred_mean = (1-1/self.N) * self.pred_mean + 1/self.N * x

def experiment(m1, m2, m3, epsilon, N):

    # Create bandits
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    # Save data for plots
    data = np.empty(N)

    # Start epsilon-greedy loops
    for i in range(N):
        # Extract epsilon
        pivot = np.random.random()
        if pivot < epsilon:
            # Take random action
            j = np.random.choice(3)
        else:
            # Take best action
            j = np.argmax([b.pred_mean for b in bandits])
        # Perform action, update mean
        x = bandits[j].pull()
        bandits[j].update(x)
        # Add data for plot
        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.pred_mean)

    return cumulative_average

c_1 = experiment(1.0, 2.0, 3.0, 0.1, 100000)
c_05 = experiment(1.0, 2.0, 3.0, 0.05, 100000)
c_01 = experiment(1.0, 2.0, 3.0, 0.01, 100000)

# log scale plot
plt.plot(c_1, label='eps = 0.1')
plt.plot(c_05, label='eps = 0.05')
plt.plot(c_01, label='eps = 0.01')
plt.legend()
plt.xscale('log')
plt.show()

# linear plot
plt.plot(c_1, label='eps = 0.1')
plt.plot(c_05, label='eps = 0.05')
plt.plot(c_01, label='eps = 0.01')
plt.legend()
plt.show()
