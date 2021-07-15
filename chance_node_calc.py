import os
from game import Game, convert_move, find_target_index
import numpy as np
import xml.etree.ElementTree as ET
from Basic_Dicts import moved


def convert(indx):
    x = int(indx / 10)
    y = indx % 10
    return x, y


def pkl_from_xml(filepath: str, final_only=False, finalize=True):
    """

    :param filepath: path to the xml file.
    :param final_only: return the game only if a player won (not surrendered).
    :param finalize: if a game is surrendered place the flag to the graveyard.
    :return: game
    """
    sums = 0
    prob = 0
    new = 0
    positives = 0

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
        # ===============================================================================================
        test_moves = game.legal_actions()
        for mv in test_moves:
            r, c, tr, tc = convert_move(mv)
            new += 1
            if game.opboard[tr][tc] == moved["hidden"] or game.opboard[tr][tc] == moved["moved"]:
                positives += 1
        prob = np.divide(prob * sums + positives, sums + new)
        sums = sums + new
        new = 0
        positives = 0
        # ===============================================================================================
        if action not in game.legal_actions():
            print("ERROR:Move not detected")
            print(elm.attrib["id"])
            print(filepath)
            print(action, r, c, tr, tc)
            print(game.state)
            for a in game.legal_actions():
                print(convert_move(a))
            break
        game.apply(action)
    winner = root.find(".//result")
    win = winner.attrib["type"]
    wplayer = winner.attrib["winner"]
    print(win, wplayer)


if __name__ == "__main__":
    pcs = {
        'A': 0., 'B': -0.5, 'C': 0.1, 'D': 0.2, 'E': 0.3, 'F': 0.4, 'G': 0.5, 'H': 0.6, 'I': 0.7, 'J': 0.8, 'K': 0.9,
        'L': 1., 'M': -0.1, 'N': -0.5, 'O': 0.1, 'P': 0.2, 'Q': 0.3, 'R': 0.4, 'S': 0.5, 'T': 0.6, 'U': 0.7, 'V': 0.8,
        'W': 0.9, 'X': 1., 'Y': -0.1, '_': 0.

    }
    column = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9

              }
    path2 = "D:\\Diploma\\games\\unpacked"
    sums = 0
    prob = 0
    new = 0
    positives = 0
    STEPS = 0
    for entry in os.listdir(path2):
        print("Starting processing folder:", os.path.join(path2, entry))
        folder = os.path.join(path2, entry)
        for file in os.listdir(os.path.join(path2, entry)):
            desc, rest = file.split('-', 1)
            if desc == 'classic':
                filepath = os.path.join(folder, file)
                tree = ET.parse(filepath)
                root = tree.getroot()
                if root.find(".//move[@id='1']") is None:
                    continue
                if root.find(".//field") is None:
                    continue
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
                    # ===============================================================================================
                    test_moves = game.legal_actions()
                    for mv in test_moves:
                        r, c, tr, tc = convert_move(mv)
                        new += 1
                        if game.opboard[tr][tc] == moved["hidden"] or game.opboard[tr][tc] == moved["moved"]:
                            positives += 1
                    prob = np.divide(prob * sums + positives, sums + new)
                    # print('temps ',new,positives)
                    sums = sums + new
                    new = 0
                    positives = 0
                    # ===============================================================================================
                    if action not in game.legal_actions():
                        print("ERROR:Move not detected")
                        # print(elm.attrib["id"])
                        # print(filepath)
                        # print(action, r, c, tr, tc)
                        # print(game.state)
                        # for a in game.legal_actions():
                        # print(convert_move(a))
                        break
                    game.apply(action)
                winner = root.find(".//result")
                win = winner.attrib["type"]
                wplayer = winner.attrib["winner"]

                print(prob, sums)
                STEPS += 1
                if STEPS >= 350:
                    break
        if STEPS >= 350:
            break
#0.055167638945614134 1373577