import math
import os
import pickle
import socket
import time
import multiprocessing as mp
import tensorflow as tf
import numpy as np

from Basic_Dicts import Pieces
from game import all_legal_actions
from q_learning.helpers import write_global_step, get_global_step

"""
Read moves from random games
create input 
create input from next states
forward pass tnetwork
get max 
create target (maybe forward pass with initial input once)
train input, target 
"""
LOCATION = r"E:\Qgames"
BATCH_SIZE = 512
gama = 0.99
terminal_values = [1, -1]
# ------------------------------------------------------
PORT = 5050
DISCONECTION = "!dis"
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def send(msg):
    message = msg.encode("utf-8")
    mlen = len(message)
    head = str(mlen).encode("utf-8")
    head += b' ' * (64 - len(head))
    client.send(head)
    client.send(message)


def rewards(g, idx):
    """
    :param g: Game
    :param idx: index of move in history
    :return: reward
    """
    assert idx < len(g.history)
    if Pieces["Flag"] not in g.graveyard:
        return 0
    reward = 0
    cd = len(g.history) - 1 - idx
    if cd < 10:
        reward = 0.15 * ((10 - cd) / 10) * g.terminal_value(idx % 2)
    return reward


def get_human_games():
    loc = r"E:\Hgames"
    games = []
    for entry in os.listdir(loc):
        games.append(entry)
    print(games[-1], len(games))
    return games


def supervised_sample(available, size: int):
    gms = math.ceil(np.divide(size, 30))
    loc = r"E:\Hgames"
    sample = np.empty([size, 10, 10, 98])
    sample_target = np.zeros([size, 1656])
    global_i = 0
    for _ in range(gms):
        idx = np.random.choice(len(available), 1)[0]
        candidate = os.path.join(loc, available[idx])
        with open(candidate, "rb") as f:
            game = pickle.load(f)
        while len(game.history) < 30:
            idx = np.random.choice(len(available), 1)[0]
            candidate = os.path.join(loc, available[idx])
            with open(candidate, "rb") as f:
                game = pickle.load(f)
        for move in range(30):
            if global_i < len(sample):
                tmp = game.make_image(move, formatted=True)
                mv = game.history[move]
                sample[global_i] = tmp
                sample_target[global_i][mv] = 1
                global_i += 1
            else:
                break
    return sample, sample_target


def generate_random_games(available, bs=BATCH_SIZE, multiple=4):
    """
    :param available: number of available games to chose from
    :param bs: batch size to be produced
    :param multiple: minimum sampled moves from a single game
    :return: a dict {game number: moves to be sampled}
    """
    if np.mod(bs, multiple) == 0 and multiple != 1:
        games = np.sort(np.random.choice(available, size=int(np.divide(bs, multiple))))
        games_dict = {}
        for v in games:
            if v not in games_dict.keys():
                games_dict[v] = multiple
            else:
                games_dict[v] += multiple
        return games_dict
    else:
        games = np.sort(np.random.choice(available, size=bs))
        games_dict = {}
        for v in games:
            if v not in games_dict.keys():
                games_dict[v] = 1
            else:
                games_dict[v] += 1
        return games_dict


def draft(g_dict):
    """
    !!!TO DO if game has fewer moves than we want, done

    :param g_dict:
    :return: states, next states, moves, rewards
    """
    location = LOCATION
    _states = np.empty([BATCH_SIZE, 10, 10, 98])
    _next = np.empty([BATCH_SIZE, 10, 10, 98])
    _moves = np.empty([BATCH_SIZE], dtype=int)
    _rewards = np.zeros([BATCH_SIZE])
    _filter = []
    failsafe = {}
    slot = 0
    for key, val in g_dict.items():
        ga = "game" + str(key) + ".pkl"
        gamepath = os.path.join(location, ga)
        try:
            with open(gamepath, 'rb') as f:
                game = pickle.load(f)

            while len(game.history) - 20 < val:
                replacement_game = np.random.choice(max(g_dict.keys()), size=1)[0]
                ga = "game" + str(replacement_game) + ".pkl"
                gamepath = os.path.join(location, ga)
                try:
                    with open(gamepath, 'rb') as f:
                        game = pickle.load(f)
                except EOFError:
                    continue

            """
            # THIS BLOCKS MOVES FROM FIXED OP TO GET DRAFTED IF WE WANT TO TRAIN ONLY FROM POLICY NETWORK
            try:
                fixed = game.q_fixed
            except AttributeError:
                fixed = None

            if fixed is None:
                state_positions = np.random.choice(len(game.history), size=val, replace=False)
            else:
                assert fixed in [0, 1]  # DELETE
                valid_states = np.arange(1 - fixed, len(game.history), 2)
                state_positions = np.random.choice(valid_states, size=val, replace=False)
            """
            state_positions = np.random.choice(np.arange(start=15, stop=len(game.history)), size=val, replace=False)
            #  ## MODIFY PROB
            for idx in range(len(state_positions)):
                choice = np.random.choice(2, 1, p=[0.93, 0.07])[0]
                if choice == 1:
                    state_positions[idx] = np.random.randint(low=len(game.history) - 10, high=len(game.history))
            ###
            for idx in state_positions:
                tmp = game.make_image(idx, formatted=True)
                _states[slot] = tmp

                if idx != len(game.history) - 1 and idx != len(game.history) - 2:
                    tmp = game.make_image(idx + 2, formatted=True)
                    _filter.append(all_legal_actions(game.state_history[idx + 2]))
                    if len(game.history) - 1 - idx < 10:  # the 9 previous moves before last
                        _rewards[slot] = rewards(game, idx)
                else:
                    tmp = game.make_image(-1, formatted=True)
                    _rewards[slot] = game.terminal_value(to_play=idx % 2)
                    _filter.append([0])
                _next[slot] = tmp
                _moves[slot] = game.history[idx]
                slot += 1

        except EOFError as err:
            failsafe[key] = val
            print(err)
    if failsafe:
        print(f"Cant open {failsafe}, (index:{slot}). Skipping...")
        return None
    if slot < BATCH_SIZE:
        print(f"not fully filled: last index {slot}")
        return None
    return _states, _next, _moves, _rewards, _filter


def short_draft(g_dict):
    """
    !!!TO DO if game has fewer moves than we want, done

    :param g_dict:
    :return: states, next states, moves, rewards
    """
    location = LOCATION
    _states = np.empty([BATCH_SIZE, 10, 10, 98])
    _next = np.empty([BATCH_SIZE, 10, 10, 98])
    _moves = np.empty([BATCH_SIZE], dtype=int)
    _rewards = np.zeros([BATCH_SIZE])
    _filter = []
    failsafe = {}
    slot = 0
    for key, val in g_dict.items():
        ga = "game" + str(key) + ".pkl"
        gamepath = os.path.join(location, ga)
        try:
            with open(gamepath, 'rb') as f:
                game = pickle.load(f)

            while len(game.history) - 20 < val:
                replacement_game = np.random.choice(max(g_dict.keys()), size=1)[0]
                ga = "game" + str(replacement_game) + ".pkl"
                gamepath = os.path.join(location, ga)
                try:
                    with open(gamepath, 'rb') as f:
                        game = pickle.load(f)
                except EOFError:
                    continue

            """
            # THIS BLOCKS MOVES FROM FIXED OP TO GET DRAFTED IF WE WANT TO TRAIN ONLY FROM POLICY NETWORK
            try:
                fixed = game.q_fixed
            except AttributeError:
                fixed = None

            if fixed is None:
                state_positions = np.random.choice(len(game.history), size=val, replace=False)
            else:
                assert fixed in [0, 1]  # DELETE
                valid_states = np.arange(1 - fixed, len(game.history), 2)
                state_positions = np.random.choice(valid_states, size=val, replace=False)
            """
            state_positions = np.random.choice(np.arange(start=0, stop=len(game.history)), size=val, replace=False)
            #  ## MODIFY PROB
            for idx in range(len(state_positions)):
                choice = np.random.choice(2, 1, p=[0.9, 0.1])[0]
                if choice == 1:
                    state_positions[idx] = np.random.randint(low=len(game.history) - 10, high=len(game.history))
            ###
            for idx in state_positions:
                try:
                    tmp = np.copy(game.image_history[idx])
                except IndexError:
                    tmp = game.make_image(idx, formatted=True)
                except AttributeError:
                    tmp = game.make_image(idx, formatted=True)
                _states[slot] = tmp

                if idx != len(game.history) - 1 and idx != len(game.history) - 2:
                    try:
                        tmp = np.copy(game.image_history[idx + 2])
                    except IndexError:
                        tmp = game.make_image(idx + 2, formatted=True)
                    except AttributeError:
                        tmp = game.make_image(idx + 2, formatted=True)
                    _filter.append(all_legal_actions(game.state_history[idx + 2]))
                    if len(game.history) - 1 - idx < 10:  # the 9 previous moves before last
                        _rewards[slot] = rewards(game, idx)
                else:
                    tmp = game.make_image(-1, formatted=True)
                    _rewards[slot] = game.terminal_value(to_play=idx % 2)
                    _filter.append([0])
                _next[slot] = tmp
                _moves[slot] = game.history[idx]
                slot += 1

        except EOFError as err:
            failsafe[key] = val
            print(err)
    if failsafe:
        print(f"Cant open {failsafe}, (index:{slot}). Skipping...")
        return None
    if slot < BATCH_SIZE:
        print(f"not fully filled: last index {slot}")
        return None
    return _states, _next, _moves, _rewards, _filter


def deposit(all_g, qu):
    targets = generate_random_games(all_g)
    res = draft(targets)  # short_draft(targets)
    if res:
        qu.put(res)


def perpetual_deposit(all_g, qu):
    while True:
        deposit(all_g, qu)


# ------------------------------------------------------------------------------------------------------

POLICY_NET = "location to the policy network"
TARGET_NET = "location to the policy network"
PWEIGHTS = "location to policy network weights"
TWEIGHTS = "location to target network weights"
CHECKPOINT_PATH = "path to save checkpoints"
if __name__ == "__main__":
    total = 0
    for file in os.listdir(LOCATION):
        if file[:4] == "game" and file[-4:] == ".pkl":
            total += 1
    input_queue = mp.Queue(20)

    targetmodel = tf.keras.models.load_model(POLICY_NET)
    policymodel = tf.keras.models.load_model(TARGET_NET)
    processes = []
    for _ in range(8):
        p = mp.Process(target=perpetual_deposit, args=(total, input_queue))
        processes.append(p)
    for pr in processes:
        time.sleep(2)
        pr.start()

    term = 0
    global_step = get_global_step()
    print(f"starting from {global_step}")
    while True:
        # --------------------------------------------------------------------------------------------
        states, next_states, moves, rewardss, filt = input_queue.get()
        initial_target = policymodel.predict(states)
        print("loaded from q")
        q_state = targetmodel.predict(next_states)
        max_q = np.zeros([BATCH_SIZE])

        for index in range(BATCH_SIZE):
            temp = max(q_state[index][valid] for valid in filt[index])
            max_q[index] = temp
        for index in range(BATCH_SIZE):
            if rewardss[index] in terminal_values:
                replacement = rewardss[index]
            else:
                replacement = rewardss[index] + gama * max_q[index]
            initial_target[index][moves[index]] = max(-1.0, min(replacement, 1.0))  # min(replacement, 1.0)
        # --------------------------------------------------------------------------------------------
        # T R A I N I N G
        print("training step...")
        policymodel.train_on_batch(x=states, y=initial_target)
        global_step += 1
        print("success ", global_step)

        # SAVING POLICY:
        if np.mod(global_step, 500) == 0:
            print(f"Saving policy network at step: {global_step}... ")
            policymodel.save(POLICY_NET)
            policymodel.save_weights(PWEIGHTS)
            write_global_step(global_step)
        # CREATING CHECKPOINT
        if np.mod(global_step, 5000) == 0:
            #  checkpoint_path = get_path()
            checkpoint_path = os.path.join(CHECKPOINT_PATH, "checkpoint" + str(global_step),
                                           "checkpoint" + str(global_step))
            policymodel.save_weights(checkpoint_path)

        # TRANSFERRING POLICY NETWORK TO TARGET + SAVING TARGET
        if np.mod(global_step, 500) == 0:
            print(f"Transfering policy weights to target at step {global_step}...")
            targetmodel.set_weights(policymodel.get_weights())
            targetmodel.save(TARGET_NET)
            targetmodel.save_weights(TWEIGHTS)

        # GIVE ORDER TO GENERATE GAMES
        if np.mod(global_step, 200) == 0:
            m = 'play'
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.connect(ADDR)
                send(m)
                send(DISCONECTION)
            except ConnectionRefusedError as e:
                print("eeerrr", e)
        """
        # Pause after specific number of steps
        
        if np.mod(global_step, 200000) == 0:
            inp = input("x=stop")
            if inp == "x":
                break
        """