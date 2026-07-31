from model import ResNet
from game import ConnectFour
from alpha_mcts import MCTS


class AlphaZero:
    def __init__(self, model: ResNet, optimizer, game: ConnectFour, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        pass

    def train(self, memory):
        pass

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for self.