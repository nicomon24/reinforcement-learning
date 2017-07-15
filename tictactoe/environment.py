'''
    Define the TicTacToe environment
'''
import numpy as np
import math, time

class TicTacToe():

    def __init__(self):
        # Create the board
        self.board = np.zeros((3,3))

    def reset(self):
        self.board = np.zeros((3,3))

    # ------ HELPERS --------

    # Print the board, use numpy print as default
    def _print(self):
        print(self.board)

    @staticmethod
    def hash_state(board):
        mask = np.array([3**i for i in range(9)])
        return int(sum(mask * board.flatten()))

    # Get an hash of the state as base3 number
    def get_state(self):
        return self.hash_state(self.board)

    # Return action from 0 to 8
    def action_space(self):
        return np.where( self.board.flatten() == 0)[0]

    # Take action from 0 to 8
    def take_action(self, player, action):
        if self.board[action // 3, action % 3] != 0:
            return False
        else:
            self.board[action // 3, action % 3] = player
            return True

    # Return the state that is reached if action i is performed
    def next_state(self, player, action):
        next_board = np.copy(self.board)
        next_board[action // 3, action % 3] = player
        return self.hash_state(next_board)

    # Game end:
    # 1-2: player that wins
    # 0: game has not ended
    # -1: draw
    def winner(self):
        # Check both winners
        for p in range(1,3):
            indexes = np.where(self.board == p)
            for i in range(3):
                # Check rows
                if len([x for x in indexes[0] if x == i]) == 3:
                    return p
                # Check cols
                if len([x for x in indexes[1] if x == i]) == 3:
                    return p
            # Check normal diagonal
            if len([x for x in self.board.diagonal() if x == p]) == 3:
                return p
            # Check inverse diagonal
            if len([x for x in np.fliplr(self.board).diagonal() if x == p]) == 3:
                return p
        # Check draw
        if len(self.action_space()) == 0:
            return -1
        # Return 0 otherwise
        return 0
