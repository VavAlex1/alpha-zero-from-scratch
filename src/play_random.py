import numpy as np

from connect_four import ConnectFour


def print_board(state):
    symbols = {1: "X", -1: "O", 0: "."}
    for row in state:
        print(" ".join(symbols[cell] for cell in row))
    print(" ".join(str(i) for i in range(state.shape[1])))


def main():
    game = ConnectFour()
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
            action = np.random.choice(np.where(valid_moves == 1)[0])
            print(f"random agent ({player}) plays {action}")

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
