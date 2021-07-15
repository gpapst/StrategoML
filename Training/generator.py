import os
import random
import numpy as np
import tensorflow as tf
import shutil

PATH = "E:\\games"
TESTPATH = "E:\\test_games"


def count_files(fpath=PATH, batch_size=1024):
    games = []

    for entry in os.listdir(fpath):
        games.append(entry)
    print(f"files: {len(games)}\n")
    length = 0
    n = 0
    for game in games:
        g = np.load(os.path.join(fpath, game))
        # gam = g["image"]
        # print(np.shape(gam))
        length += len(g["image"])
        n += 1
        if n % 400 == 0:
            print("Files counted: ", n)
    print(f"length: {length}")
    effective_length = np.ceil(length / float(batch_size))
    print(f"Number of steps per epoch with batch size {batch_size}: {effective_length}")
    return length


def split_files(origin=PATH, target=TESTPATH, percentage=5):
    flies = 0
    for file in os.listdir(origin):
        rn = np.random.randint(0, 100)
        if rn <= percentage:
            flies += 1
            shutil.move(os.path.join(origin, file), os.path.join(target, file))
    print(f"Moved {flies} files to: {target}")


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size, length, fpath=PATH):
        self.batch_size = batch_size

        self.fpath = fpath
        self.games = []
        self.current_game = 1
        for entry in os.listdir(fpath):
            self.games.append(entry)
        random.shuffle(self.games)
        self.length = length
        gam = np.load(os.path.join(self.fpath, self.games[0]))
        self.image = gam["image"]
        self.moves = gam["moves"]
        self.values = gam["values"]
        self.initialize()
        self.shuffle()

    def __len__(self):
        return int(np.ceil(self.length / float(self.batch_size)))

    def initialize(self):
        while len(self.image) <= self.batch_size:
            gam = np.load(os.path.join(self.fpath, self.games[self.current_game]))
            self.image = np.concatenate((self.image, gam["image"]))
            self.moves = np.concatenate((self.moves, gam["moves"]))
            self.values = np.concatenate((self.values, gam["values"]))
            self.current_game += 1
            if self.current_game >= len(self.games):
                random.shuffle(self.games)
                self.current_game = 0

    def shuffle(self):
        shuffler = np.random.permutation(len(self.image))
        self.image = self.image[shuffler]
        self.moves = self.moves[shuffler]
        self.values = self.values[shuffler]

    def __getitem__(self, idx):
        if self.current_game >= len(self.games):
            self.on_epoch_end()
        if len(self.image) <= self.batch_size:
            self.initialize()

        image_batch = self.image[:self.batch_size]
        moves_batch = self.moves[:self.batch_size]
        values_batch = self.values[:self.batch_size]
        self.values = self.values[self.batch_size:]
        self.moves = self.moves[self.batch_size:]
        self.image = self.image[self.batch_size:]

        return image_batch, moves_batch, [None]

    def on_epoch_end(self):
        random.shuffle(self.games)
        gam = np.load(os.path.join(self.fpath, self.games[0]))
        self.current_game = 1
        self.image = gam["image"]
        self.moves = gam["moves"]
        self.values = gam["values"]
        self.initialize()
        self.shuffle()


class WDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size, length, fpath="E:\\winner_only"):
        self.batch_size = batch_size

        self.fpath = fpath
        self.games = []
        self.current_game = 1
        for entry in os.listdir(fpath):
            self.games.append(entry)

        self.length = length
        gam = np.load(os.path.join(self.fpath, self.games[0]))
        self.image = gam["image"]
        self.moves = gam["moves"]

        self.initialize()

    def __len__(self):
        return self.length

    def initialize(self):
        while len(self.image) <= self.batch_size:
            gam = np.load(os.path.join(self.fpath, self.games[self.current_game]))
            self.image = np.concatenate((self.image, gam["image"]))
            self.moves = np.concatenate((self.moves, gam["moves"]))

            self.current_game += 1
            if self.current_game >= len(self.games) - 5:
                random.shuffle(self.games)
                self.current_game = 0
        self.shuffle()

    def shuffle(self):
        shuffler = np.random.permutation(len(self.image))
        self.image = self.image[shuffler]
        self.moves = self.moves[shuffler]

    def __getitem__(self, idx):
        if self.current_game >= len(self.games):
            self.on_epoch_end()
        if len(self.image) <= self.batch_size:
            self.initialize()

        image_batch = self.image[:self.batch_size]
        moves_batch = self.moves[:self.batch_size]
        self.moves = self.moves[self.batch_size:]
        self.image = self.image[self.batch_size:]

        return image_batch, moves_batch, [None]

    def on_epoch_end(self):
        random.shuffle(self.games)
        gam = np.load(os.path.join(self.fpath, self.games[0]))
        self.current_game = 1
        self.image = gam["image"]
        self.moves = gam["moves"]
        self.initialize()

