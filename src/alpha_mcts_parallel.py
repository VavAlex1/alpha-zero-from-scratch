import math
import numpy as np
from game import ConnectFour
from model import ResNet
import torch

class Node:
    def __init__(self, game: ConnectFour, args, state, parent=None, action_taken=None, prior=0.0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = 0
        self.value_sum = 0
    
    def is_fully_expanded(self):
        return len(self.children) > 0

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - (child.value_sum / child.visit_count + 1) / 2
        return q_value + self.args["C"] * child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1) 

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(
                    game=self.game,
                    args=self.args,
                    state=child_state,
                    parent = self,
                    action_taken=action,
                    prior=prob
                )
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent:
            self.parent.backpropagate(value)

class MCTSParallel:
    def __init__(self, game: ConnectFour, args: dict, model: ResNet):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state)
        device = self.model.device

        for _ in range(self.args["num_searches"]):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                encoded_state = self.game.get_encoded_state(node.state)
                policy, value = self.model(
                    torch.tensor(encoded_state, device=device).unsqueeze(0)
                )

                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs



