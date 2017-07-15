'''
    This is the main game loop
    Allows Player vs. Agent or Agent vs. Agent
'''
import numpy as np
import math, time
from environment import *
from player import *
from agent import *

from IPython import embed

# Create the players
a1 = Agent(1)
a2 = Agent(2)

# Define the environment
env = TicTacToe()

players = [a1, a2]
for episode in range(10000):

    if episode % 300 == 0:
        print(episode)

    env.reset()

    current_player_index = 0
    # Loop while the game is not finished
    while env.winner() == 0:
        # Print board
        #print("BOARD:\n")
        #env._print()
        # Ask player what to do
        current_player = players[current_player_index]
        played = current_player.play(env, verbose=False)
        # Check that the action was correct
        if played:
            current_player_index = (current_player_index + 1) % 2

    # Assign reward when done
    if env.winner() == -1:
        players[0].assign_reward(0.1)
        players[1].assign_reward(0.1)
    else:
        players[(env.winner()+1) % 2].assign_reward(1)
        players[(env.winner()) % 2].assign_reward(0)

    #print("WINNER: ", env.winner())
    #print("BOARD:\n")
    #env._print()


players = [Player(1), a2]
for episode in range(100):

    # Define the environment
    env = TicTacToe()

    current_player_index = 0
    # Loop while the game is not finished
    while env.winner() == 0:
        # Print board
        print("BOARD:\n")
        env._print()
        # Ask player what to do
        current_player = players[current_player_index]
        played = current_player.play(env)
        # Check that the action was correct
        if played:
            current_player_index = (current_player_index + 1) % 2

    # Assign reward when done
    if env.winner() == -1:
        players[0].assign_reward(0.1)
        players[1].assign_reward(0.1)
    else:
        players[(env.winner()+1) % 2].assign_reward(1)
        players[(env.winner()) % 2].assign_reward(0)

    print("WINNER: ", env.winner())
    print("BOARD:\n")
    env._print()
