import json
import os
import pickle
from q_learning.qnet import QResNet
import numpy as np
import collections
import shutil

MAX_SIZE = 10000
LOCATION = r"E:\Qgames"
AVERAGE = r'running_avg'
GLOBAL_STEP = r'global.txt'


def move(root_src_dir, root_dst_dir):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # in case of the src and dst are the same file
                if os.path.samefile(src_file, dst_file):
                    continue
                os.remove(dst_file)
            shutil.move(src_file, dst_dir)


root_src_dir = r'E:\Qfiles\TEST\net2'
root_dst_dir = r'E:\Qfiles\TEST\net1'


# move(root_src_dir, root_dst_dir)
# shutil.rmtree(root_src_dir)

def get_path():
    target = r'E:\Qfiles\CHECKPOINTS'
    check = 0
    for drctr in os.listdir(target):
        if drctr[:10] == "checkpoint":
            _, n = drctr.split('checkpoint', 1)
            if int(n) > check:
                check = int(n)

    new = os.path.join(target, "checkpoint" + str(check + 1))
    return new


def running_average(new: bool):
    max_len = 50
    try:
        with open(AVERAGE, 'rb') as file:
            running_avg = pickle.load(file)
    except FileNotFoundError:
        running_avg = []
    if running_avg:
        if len(running_avg) >= max_len:
            running_avg.pop(0)
        running_avg.append(new)
        terminal = running_avg.count(True)
        percentage = np.divide(terminal, len(running_avg))
    else:
        running_avg.append(new)
        percentage = 0
    with open(AVERAGE, 'wb') as file:
        pickle.dump(running_avg, file)
    return percentage


def get_global_step():
    with open(GLOBAL_STEP) as json_file:
        data = json.load(json_file)
    return data["global"]


def write_global_step(glob):
    with open(GLOBAL_STEP) as json_file:
        data = json.load(json_file)
    data["global"] = glob
    with open(GLOBAL_STEP, 'w') as json_file:
        json.dump(data, json_file)
    return


def save_game(game, *, index: collections.deque):
    """

    :param game:
    :param index:
    :return:
    """
    game.prepare_images()
    location = LOCATION
    if len(index) == 0:
        name = "game" + str(0) + ".pkl"
        with open(os.path.join(location, name), "wb") as file:
            pickle.dump(game, file)
        index.append(name)
    elif len(index) == index.maxlen:
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

