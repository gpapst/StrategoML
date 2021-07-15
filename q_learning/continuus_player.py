import sys
import time

# sys.path.append("....")

from q_learning.helpers import save_game, get_index, running_average
from q_learning.remote_controlled_player import player_worker
import tensorflow as tf
import numpy as np

opponent = r"E:\Qfiles\active_op"
Network = r'E:\Qfiles\policy_net'
Weights = r"E:\Qfiles\pweights\weights"

print("init...")
net = tf.keras.models.load_model(Network)
opp = tf.keras.models.load_model(opponent)
index = get_index()
step = 0
print("playing games...")
while True:
    time.sleep(1)
    step += 1

    g = player_worker(net, opp)
    pcnt = running_average(new=g.terminal())
    if g.terminal():
        if len(g.history) > 30:
            save_game(g, index=index)

            print("delivered")
    if np.mod(step, 50) == 0:
        net.load_weights(Weights)
        print(pcnt)
