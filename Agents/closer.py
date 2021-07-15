from Basic_Dicts import Pieces, moved
from game import Game, find_target_index, convert_move
import numpy as np
import tensorflow as tf
from preprocess import pkl_from_xml


def evaluate(game: Game):
    info = {
        "own_top_loc": [],
        "own_miner_loc": [],
        "own_top_id": 0,
        "own_top": 0,
        "op_moved_locs": [],
        "op_top_id": 0,
        "op_top": 0,
        "op_miners": 0,
        "op_bombs": 0,
        "own_miners": 0,
        "op_hidden": 0,
        "all_locs": [],

    }
    for x, y in np.ndindex(10, 10):
        if game.board[x][y] != 0:
            if game.opboard[x][y] != 0:
                if game.board[x][y] == Pieces["Miner"]:
                    info["op_miners"] += 1
                if game.board[x][y] > info["op_top_id"]:
                    info["op_top_id"] = game.board[x][y]
                    info["op_top"] = 1

                elif game.board[x][y] == info["op_top_id"]:
                    info["op_top"] += 1
                if game.opboard[x][y] == game.board[x][y] or game.opboard[x][y] == moved["moved"]:
                    info["op_moved_locs"].append((x, y))

                if game.opboard[x][y] == moved["hidden"]:
                    info["op_hidden"] += 1
            if game.ownboard[x][y] != 0:
                if game.board[x][y] == Pieces["Miner"]:
                    info["own_miners"] += 1
                    info["own_miner_loc"].append((x, y))
                if game.board[x][y] > info["own_top_id"]:
                    info["own_top_id"] = game.board[x][y]
                    info["own_top"] = 1
                    info["own_top_loc"] = [(x, y)]
                elif game.board[x][y] == info["op_top_id"]:
                    info["own_top"] += 1
                    info["own_top_loc"].append((x, y))

    if info["own_top_id"] == info["op_top_id"] and info["own_top"] > info["op_top"]:
        info["all_locs"] = [pos for pos in info["own_top_loc"]]
        return info
    if info["own_top_id"] > info["op_top_id"]:
        if info["own_top_id"] > info["op_top_id"] + 0.1:
            for x, y in np.ndindex(10, 10):
                if game.board[x][y] > info["op_top_id"] and game.ownboard[x][y] != 0:
                    info["all_locs"].append((x, y))
        return info

    return None


def pathing(g: Game, *args, start, destinations):
    lake = [(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)]
    initial_grid = np.zeros((10, 10)) + 100
    booster = {}
    for sx, sy in start:
        starting_tile = (sx, sy)
        grid = np.copy(initial_grid)
        grid[sx][sy] = 0
        visited = [(sx, sy)]
        t1 = [(x, y) for x, y in np.ndindex(2, 2)] + [(-x, -y) for x, y in np.ndindex(2, 2)]
        t2 = [(sx + x, sy + y) for x, y in t1 if x != y]
        limit = [(x, y) for (x, y) in t2 if (0 <= x <= 9 and 9 >= y >= 0)]

        dist = 1
        new_limit = []
        while limit:
            for pos in limit:
                (x, y) = pos
                if g.board[x][y] == 0:
                    if (x, y) not in lake:
                        grid[x][y] = dist

                visited.append(pos)
            for pos in limit:
                (x, y) = pos
                if grid[x][y] != 100:
                    sx, sy = pos
                    t1 = [(x, y) for x, y in np.ndindex(2, 2)] + [(-x, -y) for x, y in np.ndindex(2, 2)]
                    t2 = [(sx + x, sy + y) for x, y in t1 if x != y]
                    new_limit = list(set(new_limit) | set([(x, y) for (x, y) in t2 if (0 <= x <= 9 and 9 >= y >= 0)]))

                new_limit = [p for p in new_limit if p not in visited]

            # print(limit)
            limit = new_limit
            new_limit = []
            dist += 1
        # print("\n", grid)
        for dx, dy in destinations:
            paths = [(dx, dy)]
            while True:
                t1 = [(x, y) for x, y in np.ndindex(2, 2)] + [(-x, -y) for x, y in np.ndindex(2, 2)]
                t2 = [(dx + x, dy + y) for x, y in t1 if x != y]
                lim = [(x, y) for (x, y) in t2 if (0 <= x <= 9 and 9 >= y >= 0)]
                lim_dict = {(x, y): grid[x][y] for (x, y) in lim}
                min_val = min(lim_dict.values())
                candidates = [k for k, v in lim_dict.items() if v == min_val]
                next_cell = max(candidates, key=lambda t: t[0])
                if lim_dict[next_cell] < grid[dx][dy]:
                    paths.append(next_cell)
                    dx, dy = next_cell
                else:
                    break
                # print(next_cell)
                # print(lim, lim_dict)
                if lim_dict[next_cell] == 0:
                    break
            # print("path", paths)
            if paths[-1] == starting_tile:
                # print(paths[-1], "->", paths[-2])
                move = find_target_index(paths[-1][0], paths[-1][1], paths[-2][0], paths[-2][1])
                # print(move, convert_move(move))
                #  bonus if close to capture
                if len(paths) <= 3:
                    capture_bonus = 0.4
                else:
                    capture_bonus = 0
                if move not in booster.keys():
                    #  bonus if moving forward
                    fwd_bonus = 0
                    if paths[-1][0] > paths[-2][0]:
                        fwd_bonus = 0.05
                    boost = 2 - 0.1 * len(paths) + fwd_bonus + capture_bonus

                    # print("boost", boost)
                    booster[move] = boost
                else:
                    boost = (2 - 0.1 * len(paths)) * 0.5 + capture_bonus
                    # print("boost", boost)
                    booster[move] += boost
            # print(booster)

    return booster


def closer(game: Game):
    """
    Evaluates if the Game is in an obviously winning position and proposes moves for certain victory if possible.
    :param game: Game
    :return: dictionary of proposed moves and their relative value.
    """

    result = evaluate(game)
    if result:
        boosted_moves = pathing(game, start=result["all_locs"], destinations=result["op_moved_locs"])
        return boosted_moves
    else:
        return {}


# g1 = pkl_from_xml(r"G:\Diploma\games\unpacked\strados2014-7\classic-2014.7-1.xml", finalize=False)

g1 = pkl_from_xml(r"G:\Diploma\games\unpacked\strados2014-7\classic-2014.7-1016.xml", finalize=False)

print("\n\n", closer(g1))

net = tf.keras.models.load_model(r'G:\new_CNN300005dadam\cp-0020.ckpt')
actions = g1.legal_actions()
ls = g1.to_play()
print(ls)

g1.display()

while not g1.terminal():
    if g1.to_play() == ls:
        print("playing ", g1.to_play())
        actions = g1.legal_actions()
        inputs = g1.make_image(-1, formatted=True)
        inputs = np.expand_dims(inputs, axis=0)
        values = net.predict(inputs)
        tmp = {action: values[0][action] for action in actions}
        move = max(tmp, key=tmp.get)
        print(convert_move(move))
        g1.display()
        g1.apply(move)
    else:
        print("playing ", g1.to_play())
        actions = g1.legal_actions()
        boosted = closer(g1)
        if boosted:
            move = max(boosted, key=boosted.get)
            print(convert_move(move))
            g1.display()
            g1.apply(move)
        else:
            print("empty")
            g1.display()

g1.display()
