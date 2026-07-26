import time

import numpy as np

from game import ConnectFour
from mcts import MCTS


def print_board(state):
    symbols = {1: "X", -1: "O", 0: "."}
    for row in state:
        print(" ".join(symbols[cell] for cell in row))
    print(" ".join(str(i) for i in range(state.shape[1])))


def main():
    game = ConnectFour()
    args = {
        "C": 2,
        "num_searches": 1000,
    }
    mcts = MCTS(game, args)

    state = game.get_initial_state()
    player = 1
    human_player = 1

    while True:
        print()
        print_board(state)

        valid_moves = game.get_valid_moves(state)

        if player == human_player:
            print("valid moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
            raw = input(f"player {player}: ")
            if not raw.isdigit() or int(raw) >= game.action_size or valid_moves[int(raw)] == 0:
                print("action not valid")
                continue
            action = int(raw)
        else:
            neutral_state = game.change_perspective(state, player)
            start = time.time()
            action_probs = mcts.search(neutral_state)
            elapsed = time.time() - start
            action = int(np.argmax(action_probs))
            print(f"mcts ({player}) plays {action} (thought for {elapsed:.2f}s)")

        state = game.get_next_state(state, action, player)
        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            print()
            print_board(state)
            if value == 1:
                print(f"player {player} won")
            else:
                print("draw")
            break

        player = game.get_opponent(player)


if __name__ == "__main__":
    main()
