import csv
import os

from Basic_Dicts import Pieces
from game import Game, convert_move, find_target_index
import numpy as np
import xml.etree.ElementTree as ET


def convert(indx):
    x = int(indx / 10)
    y = indx % 10
    return x, y


def preprocess_xml(filepath: str):
    """

    :param filepath: xml file
    :return:
    """
    pcs = {
        'A': 0., 'B': -0.5, 'C': 0.1, 'D': 0.2, 'E': 0.3, 'F': 0.4, 'G': 0.5, 'H': 0.6, 'I': 0.7, 'J': 0.8, 'K': 0.9,
        'L': 1., 'M': -0.1, 'N': -0.5, 'O': 0.1, 'P': 0.2, 'Q': 0.3, 'R': 0.4, 'S': 0.5, 'T': 0.6, 'U': 0.7, 'V': 0.8,
        'W': 0.9, 'X': 1., 'Y': -0.1, '_': 0.

    }
    column = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9

              }

    tree = ET.parse(filepath)
    root = tree.getroot()
    if root.find(".//move[@id='1']") is None:
        return
    if root.find(".//field") is None:
        return
    pos = root.find(".//field").attrib['content']
    reverse = False
    board = np.zeros((10, 10))
    idx = 0

    for character in pos:
        if character in pcs.keys():
            i, j = convert(idx)
            board[i][j] = pcs[character]
            idx += 1

    game = Game()

    if root.find(".//move[@id='1']").attrib['source'][1] == '4':
        b = np.flip(board, axis=1)
        board = np.flip(b, axis=0)
        reverse = True
        game.state[0][:][:] = board[:][:]

    for elm in root.findall(".//move"):
        src = elm.attrib['source']
        trg = elm.attrib['target']
        if src[1] == ':':
            r = 9
        else:
            r = int(src[1]) - 1
        c = column[src[0]]
        if trg[1] == ':':
            tr = 9
        else:
            tr = int(trg[1]) - 1
        tc = column[trg[0]]

        if reverse:
            r = 9 - r
            c = 9 - c
            tr = 9 - tr
            tc = 9 - tc
        reverse = not reverse

        action = find_target_index(r, c, tr, tc)
        if action not in game.legal_actions():
            print("ERROR:Move not detected")
            print(elm.attrib["id"])
            print(filepath)
            print(action, r, c, tr, tc)
            print(game.state)
            for a in game.legal_actions():
                print(convert_move(a))
            return
        game.apply(action)

    # --------------------------------------------------------------------------------
    # CREATING DATA

    inputs = []
    for idx in range(len(game.state_history)):
        inputs.append(game.make_image(idx, formatted=True))
    move_target = np.zeros((len(game.history), 1656))
    for index, elem in enumerate(game.history):
        move_target[index][elem] = 1
    winner = root.find(".//result")

    if winner.attrib["type"] == "1" or winner.attrib["type"] == "3" or winner.attrib["type"] == "0":
        winning_game = True
        if winner.attrib["winner"] == "1":
            values = np.empty(len(game.history))
            values[::2] = 1
            values[1::2] = -1
            values = np.reshape(values, (len(game.history), 1))
            input_winning = inputs[::2]
            moves_winning = move_target[::2]
        else:
            values = np.empty(len(game.history))
            values[::2] = -1
            values[1::2] = 1
            values = np.reshape(values, (len(game.history), 1))
            input_winning = inputs[1::2]
            moves_winning = move_target[1::2]
    else:
        winning_game = False
        values = np.zeros((len(game.history), 1))

    # --------------------------------------------------------------------------------
    # SAVING TO FILES

    target_path2 = ".\\games\\winner_only"
    target_path = ".\\games\\games"
    csv_path = "C:\\Users\\user\\PycharmProjects\\StrategoCPU\\positions.csv"
    _, name = filepath.rsplit('\\', 1)
    _, gnumber = name.split('-', 1)
    gnumber, _ = gnumber.rsplit('.', 1)
    gnumber = gnumber.replace(".", "_")
    gnumber = gnumber.replace("-", "_")
    target_path = target_path + "\\" + gnumber + ".npz"
    target_path2 = target_path2 + "\\" + gnumber + "_w.npz"
    inputs = np.asarray(inputs)

    np.savez(target_path, image=inputs, moves=move_target, values=values)
    if winning_game:
        input_winning = np.asarray(input_winning)
        np.savez(target_path2, image=input_winning, moves=moves_winning)

    with open(csv_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([pos])


def pkl_from_xml(filepath: str, final_only=False, finalize=True):
    """

    :param filepath: path to the xml file.
    :param final_only: return the game only if a player won (not surrendered).
    :param finalize: if a game is surrendered place the flag to the graveyard.
    :return: game
    """

    # 1: capture, 0: no moves, 3:ff, 4:player did not move
    pcs = {
        'A': 0., 'B': -0.5, 'C': 0.1, 'D': 0.2, 'E': 0.3, 'F': 0.4, 'G': 0.5, 'H': 0.6, 'I': 0.7, 'J': 0.8, 'K': 0.9,
        'L': 1., 'M': -0.1, 'N': -0.5, 'O': 0.1, 'P': 0.2, 'Q': 0.3, 'R': 0.4, 'S': 0.5, 'T': 0.6, 'U': 0.7, 'V': 0.8,
        'W': 0.9, 'X': 1., 'Y': -0.1, '_': 0.

    }
    column = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9

              }

    tree = ET.parse(filepath)
    root = tree.getroot()
    if root.find(".//move[@id='1']") is None:
        return
    if root.find(".//field") is None:
        return
    pos = root.find(".//field").attrib['content']
    reverse = False
    board = np.zeros((10, 10))
    idx = 0

    for character in pos:
        if character in pcs.keys():
            i, j = convert(idx)
            board[i][j] = pcs[character]
            idx += 1

    game = Game()

    if root.find(".//move[@id='1']").attrib['source'][1] == '4':
        b = np.flip(board, axis=1)
        board = np.flip(b, axis=0)
        reverse = True
        game.state[0][:][:] = board[:][:]

    for elm in root.findall(".//move"):
        src = elm.attrib['source']
        trg = elm.attrib['target']
        if src[1] == ':':
            r = 9
        else:
            r = int(src[1]) - 1
        c = column[src[0]]
        if trg[1] == ':':
            tr = 9
        else:
            tr = int(trg[1]) - 1
        tc = column[trg[0]]

        if reverse:
            r = 9 - r
            c = 9 - c
            tr = 9 - tr
            tc = 9 - tc
        reverse = not reverse

        action = find_target_index(r, c, tr, tc)
        if action not in game.legal_actions():
            print("ERROR:Move not detected")
            print(elm.attrib["id"])
            print(filepath)
            print(action, r, c, tr, tc)
            print(game.state)
            for a in game.legal_actions():
                print(convert_move(a))
            return
        game.apply(action)
    winner = root.find(".//result")
    win = winner.attrib["type"]
    wplayer = winner.attrib["winner"]
    # print(win, wplayer)
    if finalize:
        if int(win) == 3:
            if int(wplayer) == game.to_play() + 1:
                for x, y in np.ndindex(10, 10):
                    if game.board[x][y] == Pieces["Flag"] and game.opboard[x][y] != 0:
                        game.send_to_graveyard(piece=game.board[x][y], own=False)
                        game.board[x][y] = 0
                        game.opboard[x][y] = 0
                # print("winner is the player")
            else:

                for x, y in np.ndindex(10, 10):
                    if game.board[x][y] == Pieces["Flag"] and game.ownboard[x][y] != 0:
                        # print("found")

                        game.send_to_graveyard(piece=game.board[x][y], own=True)
                        game.board[x][y] = 0
                        game.ownboard[x][y] = 0
                # print("winner is the opponent")
                # print(game.graveyard)

    if final_only:
        if win == "0" or win == "1":
            return game
        else:
            return None
    # print(win)
    if win == "0" or win == "1" or win == "3":
        return game
    else:
        return None


def avg_move_finder():  # bghkan 278.9521314261436
    all = 0
    movecount = 0
    foldercount = 0
    path2 = "E:\\Diploma\\games\\unpacked"
    for entry in os.listdir(path2):
        foldercount += 1
        fold = 0
        print("Starting processing folder:", os.path.join(path2, entry))
        for file in os.listdir(os.path.join(path2, entry)):

            desc, rest = file.split('-', 1)
            if desc == 'classic':
                tp = os.path.join(path2, entry, file)
                # print(tp)
                g = pkl_from_xml(filepath=tp)
                if g:
                    movecount += len(g.history)
                    all += 1
                    fold += 1
        print(fold)
        if foldercount >= 5:
            break
    print(all)
    print(np.divide(movecount, all))
