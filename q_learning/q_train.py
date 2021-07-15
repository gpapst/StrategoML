import collections
import os
import random
import time
import numpy as np
import pickle
from game import Game
import tensorflow as tf

from q_learning.qnet import QResNet

opponent = "E:\\Qfiles\\active_op"
max_epsilon = 0.1
min_epsilon = 0.01
MAX_SIZE = 10000
LOCATION = f"E:\Qgames"


def save_game(game, *, index: collections.deque):
    """

    :param game:
    :param index:
    :return:
    """
    location = LOCATION
    if len(index) == index.maxlen:
        name = index.popleft()
        with open(os.path.join(location, name), "wb") as file:
            pickle.dump(game, file)
        index.append(name)
    else:
        temp = index[-1]
        _, no = temp.split("e", 1)
        no, _ = no.split(".", 1)
        name = "game" + str(int(no) + 1) + ".pkl"
        with open(os.path.join(location, name), "wb") as file:
            pickle.dump(game, file)
        index.append(name)
    with open(os.path.join(LOCATION, "index.pkl"), "wb") as f:
        pickle.dump(index, f)


def get_index():
    try:
        with open(os.path.join(LOCATION, "index.pkl"), "rb") as f:
            idx = pickle.load(f)
        return idx
    except FileNotFoundError:
        print("index file not found")
        idx = init_index()
    except EOFError:
        print("Cant open index file")
        idx = init_index()
    return idx


def init_index():
    q = collections.deque(maxlen=MAX_SIZE)
    lst = []
    for file in os.listdir(LOCATION):
        if file[:4] == "game" and file[-4:] == ".pkl":
            _, b = file.split("e", 1)
            b, _ = b.split(".", 1)
            lst.append(int(b))
    lst.sort()
    for no in lst:
        g = "game" + str(no) + ".pkl"
        q.append(g)
    return q


def player_worker(net, op):
    game = Game()
    fixed_player = np.random.choice([0, 1], 1)[0]
    game.q_fixed = fixed_player
    current_player = 0
    while not game.terminal():
        actions = game.legal_actions()
        if current_player == fixed_player:  # Opponent plays
            explore = np.random.choice([True, False], 1, p=[min_epsilon, 1 - min_epsilon])[0]
            if explore:
                move = np.random.choice(actions, 1)[0]
            else:
                inputs = game.make_image(-1, formatted=True)
                inputs = np.expand_dims(inputs, axis=0)
                values = op.predict(inputs)

                tmp = {action: values[0][action] for action in actions}
                move = max(tmp, key=tmp.get)
        else:
            epsilon = max((max_epsilon - np.divide((min_epsilon - min_epsilon) * len(game.history), 100), min_epsilon))
            explore = np.random.choice([True, False], 1, p=[epsilon, 1 - epsilon])[0]
            if explore:
                move = np.random.choice(actions, 1)[0]
            else:
                inputs = game.make_image(-1, formatted=True)
                inputs = np.expand_dims(inputs, axis=0)
                values = net.Model.predict(inputs)

                tmp = {action: values[0][action] for action in actions}

                move = max(tmp, key=tmp.get)
        game.apply(move)
        current_player = 1 - current_player
        print(f"move {len(game.history)}")
        if len(game.history) > 800:
            break
    return game


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

def max_move(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def play_games(conn, lock, *, q):
    game = Game()
    game.reset_file()
    step = 0
    while not game.terminal():

        img = game.make_image(-1, formatted=True)

        conn.send(img)
        res = conn.recv()
        with lock:
            print(os.getpid(), f"at step {step} got:", type(res), np.shape(res))
        valid = game.legal_actions()
        moves = {v: res[0][v] for v in valid}
        if step < 1:
            move = random.choice(list(moves.keys()))
        else:
            move = max_move(moves)
        game.apply(move)
        step += 1
        if step > 3:
            break
    q.put(game)
    print("waiting 10")
    time.sleep(10)
    conn.close()
    """
    while True:
        game=Game()
        game.reset_file()
        while not game.terminal():
    """


if __name__ == "__main__":
    net = QResNet()
    net.compile()
    opp = tf.keras.models.load_model(opponent)
    index = get_index()
    print("init ok")
    sample_game = player_worker(net, opp)
    sample_game.display()


