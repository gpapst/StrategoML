import random

import numpy as np
import tensorflow as tf

from Basic_Dicts import Pieces
from game import Game
from Agents.multiply import MultiPly


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def count_mat(state):
    base_values = {
        0: 0,
        Pieces["Flag"]: 600,
        Pieces["Marshal"]: 400,
        Pieces["General"]: 300,
        Pieces["Colonel"]: 175,
        Pieces["Major"]: 140,
        Pieces["Captain"]: 100,
        Pieces["Lieutenant"]: 50,
        Pieces["Sergeant"]: 25,
        Pieces["Miner"]: 100,
        Pieces["Scout"]: 10,
        Pieces["Spy"]: 100,
        Pieces["Bomb"]: 80
    }
    c1 = 0
    c2 = 0
    for i, j in np.ndindex(state[0].shape):
        c1 += base_values[state[0][i][j]] * bool(state[2][i][j])
        c2 += base_values[state[0][i][j]] * bool(state[1][i][j])
    print(c1, c2)
    if c1 > c2:
        return "win"
    if c2 > c1:
        return "loss"
    return "draw"


"""
EXAMPLE
p2 = tf.keras.models.load_model("NETWORK_PATH")
p1 = tf.keras.models.load_model("NETWORK_PATH")
"""

stats = {}
agent1 = "Name1"
agent2 = "Name2"
agent1_mat = "p1mat"
agent2_mat = "p2mat"
stats[agent1] = 0
stats[agent1_mat] = 0
stats[agent2] = 0
stats[agent2_mat] = 0
stats["draw"] = 0
tgn = 400
first = 1
for g in range(tgn):
    if first == 1:  # decide fist move alternately
        player = 1
        first = 2
    else:
        player = 2
        first = 1
    number_of_moves = 0

    game = Game()
    game.reset_file()
    agent = MultiPly(plies=2)
    print(f"first:{player}")
    while True:

        if player == 1:
            # Over Max Moves
            if number_of_moves >= 750:
                # game.display()
                stats["draw"] += 1
                final = count_mat(game.state)
                if final == "win":
                    stats[agent1_mat] += 1
                if final == "loss":
                    stats[agent2_mat] += 1
                break
            ######################

            choice = np.random.choice(3, 1, p=[0.8, 0.18, 0.02])[0]

            actions = game.legal_actions()
            if choice == 0:
                inputs = game.make_image(-1, formatted=True)
                inputs = np.expand_dims(inputs, axis=0)
                values = p1.predict(inputs)
                tmp = {action: values[0][action] for action in actions}
                move = max(tmp, key=tmp.get)
            elif choice == 1:
                inputs = game.make_image(-1, formatted=True)
                inputs = np.expand_dims(inputs, axis=0)
                values = p1.predict(inputs)
                tmp = {action: values[0][action] for action in actions}
                ac = [k for k in tmp.keys()]
                values = np.array([v for v in tmp.values()])  #
                values = softmax(values)
                move = np.random.choice(actions, 1, p=values)[0]
            elif choice == 2:
                move = random.choice(actions)
            #######################

            game.apply(move)
            number_of_moves += 1
            end = game.terminal_value(game.to_play())  # from opponents perspective now due to reversal in apply
            if end != 0:
                if end == 1:
                    stats[agent2] += 1
                elif end == -1:
                    stats[agent1] += 1
                break
            player = 2

        if number_of_moves % 100 == 0:
            print(number_of_moves)
            # game.display()

        if player == 2:  # ------------------------------------------------------------------------------------------

            #######################
            choice = np.random.choice(3, 1, p=[0.8, 0.18, 0.02])[0]
            actions = game.legal_actions()
            if choice == 0:
                inputs = game.make_image(-1, formatted=True)
                inputs = np.expand_dims(inputs, axis=0)
                values = p2.predict(inputs)
                tmp = {action: values[0][action] for action in actions}
                move = max(tmp, key=tmp.get)
            elif choice == 1:
                inputs = game.make_image(-1, formatted=True)
                inputs = np.expand_dims(inputs, axis=0)
                values = p2.predict(inputs)
                tmp = {action: values[0][action] for action in actions}
                ac = [k for k in tmp.keys()]
                values = np.array([v for v in tmp.values()])  #
                values = softmax(values)
                move = np.random.choice(actions, 1, p=values)[0]
            elif choice == 2:
                move = random.choice(actions)
            #######################

            game.apply(move)
            number_of_moves += 1
            end = game.terminal_value(game.to_play())  # from opponents perspective now due to reversal in apply
            if end != 0:
                if end == 1:
                    stats[agent1] += 1
                elif end == -1:
                    stats[agent2] += 1
                break
            player = 1
            if number_of_moves % 100 == 0:
                print(number_of_moves)
                # game.display()
    print(stats)
print("final:", stats)

"""
 ####################### NETWORK
            choice = np.random.choice(3, 1, p=[0.0, 0.98, 0.02])[0]
            actions = game.legal_actions()
            if choice == 0:
                inputs = game.make_image(-1, formatted=True)
                inputs = np.expand_dims(inputs, axis=0)
                values = p2.predict(inputs)
                tmp = {action: values[0][action] for action in actions}
                move = max(tmp, key=tmp.get)
            elif choice == 1:
                inputs = game.make_image(-1, formatted=True)
                inputs = np.expand_dims(inputs, axis=0)
                values = p2.predict(inputs)
                tmp = {action: values[0][action] for action in actions}
                ac = [k for k in tmp.keys()]
                values = np.array([v for v in tmp.values()])  #
                values = softmax(values)
                move = np.random.choice(actions, 1, p=values)[0]
            elif choice == 2:
                move = random.choice(actions)
 #######################
"""
# mp2
"""
            moves = agent.evaluate_moves(game.state)

            choice = np.random.choice(3, 1, p=[0.9, 0.08, 0.02])
            if choice[0] == 0:
                move = max(moves, key=(lambda key: moves[key]))
            elif choice[0] == 1:  # flatten probability distribution
                actions = [k for k in moves.keys()]
                values = np.array([v for v in moves.values()])
                values = softmax(values)
                move = np.random.choice(actions, 1, p=values)[0]
            else:
                move = random.choice(list(moves.keys()))
"""

"""
            moves = search_agent(game)
            #######################

            choice = np.random.choice(3, 1, p=[0.9, 0.1, 0.0])
            if choice == 0:
                move = max(moves, key=(lambda key: moves[key]))
            elif choice[0] == 1:  # probability distribution
                actions = [k for k in moves.keys()]
                values = np.array([v for v in moves.values()])
                values = softmax(values)
                move = np.random.choice(actions, 1, p=values)[0]
            else:
                move = random.choice(list(moves.keys()))
"""
