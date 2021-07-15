import os
import queue
import random
import socket
import sys
import threading
import time
import numpy as np

# sys.path.append("......")
from game import Game
import tensorflow as tf

from q_learning.helpers import get_index, save_game

opponent = "Path to active_op"
max_epsilon = 0.1
min_epsilon = 0.02
MAX_SIZE = 10000
LOCATION = f"E:\Qgames"


def player_worker(net, op):
    game = Game()
    from_file = np.random.choice([0, 1], 1, p=[0.98, 0.02])[0]
    if from_file:
        game.reset_file()
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
            epsilon = max((max_epsilon - np.divide(min_epsilon * len(game.history), 100), min_epsilon))  # exploration%
            explore = np.random.choice([True, False], 1, p=[epsilon, 1 - epsilon])[0]
            if explore:
                move = np.random.choice(actions, 1)[0]
            else:
                inputs = game.make_image(-1, formatted=True)
                inputs = np.expand_dims(inputs, axis=0)
                values = net.predict(inputs)

                tmp = {action: values[0][action] for action in actions}

                move = max(tmp, key=tmp.get)
        game.apply(move)
        current_player = 1 - current_player  # alternate players

        if len(game.history) > 650:
            break
    return game


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

def max_move(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def play_games(conn, lock, *, q):  # DEL
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
    print("waiting 2s")
    time.sleep(2)
    conn.close()
    """
    while True:
        game=Game()
        game.reset_file()
        while not game.terminal():
    """


PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
DISCONECTION = "!dis"


def handle_client(conn, addr):
    connected = True

    while connected:
        msg_len = conn.recv(64).decode("utf-8")
        if msg_len:
            msg_len = int(msg_len)
            msg = conn.recv(msg_len).decode("utf-8")
            print(msg)
            if msg == "play":
                instruction.put(msg)
            elif msg == DISCONECTION:
                connected = False
    conn.close()


def start():
    server.listen()
    print("Listening...")
    while True:
        conn, addr = server.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr))
        t.start()


"""
sample_game = player_worker(net, opp)
sample_game.display()
"""
PWEIGHTS=""
if __name__ == "__main__":
    net = tf.keras.models.load_model(opponent)
    opp = tf.keras.models.load_model(opponent)
    index = get_index()
    print("init ok")
    instruction = queue.Queue()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    print(f"[STARTING] server starting")
    tt = threading.Thread(target=start)
    tt.start()
    while True:
        print("Waiting instruction")
        instruct = instruction.get()
        if instruct == "play":
            print("playing games...")
            net.load_weights(PWEIGHTS)
            for _ in range(4):
                g = player_worker(net, opp)
                if g.terminal():
                    save_game(g, index=index)
            print("delivered")
