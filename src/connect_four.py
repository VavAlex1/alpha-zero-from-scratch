import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from tqdm import tqdm

torch.manual_seed(0)


class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
    
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        state = state.copy()
        column = state[:, action]
        empty = np.where(column == 0)[0][-1]
        state[empty, action] = player
        return state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        