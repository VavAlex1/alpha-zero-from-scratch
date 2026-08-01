from model import ResNet
from game import ConnectFour
from alpha_mcts import MCTS
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn.functional as F


class SPG:
    def __init__(self, game: ConnectFour):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None


class AlphaZeroParallel:
    def __init__(self, model: ResNet, optimizer, game: ConnectFour, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_state = self.game.change_perspective(states, player)

            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            action = np.random.choice(self.game.action_size, p=action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        device = self.model.device
        policy_loss_history = []
        value_loss_history = []

        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            policy_loss_history.append(policy_loss.item())
            value_loss_history.append(value_loss.item())

        return np.mean(policy_loss_history), np.mean(value_loss_history)


    def learn(self):
        for iteration in tqdm(range(self.args['num_iterations'])):
            memory = []

            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations'])):
                memory += self.selfPlay()

            self.model.train()
            epoch_bar = tqdm(range(self.args['num_epochs']), desc="train")
            for epoch in epoch_bar:
                policy_loss, value_loss = self.train(memory)
                epoch_bar.set_postfix(policy_loss=f"{policy_loss:.4f}", value_loss=f"{value_loss:.4f}")

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")


def main():
    game = ConnectFour()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet(game, 9, 128, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    args = {
        "C": 2,
        "num_iterations": 8,
        "num_selfPlay_iterations": 100,
        "num_epochs": 4,
        "num_searches": 250,
        "batch_size": 128
    }

    alphazero = AlphaZero(model, optimizer, game, args)
    alphazero.learn()

if __name__ == "__main__":
    main()