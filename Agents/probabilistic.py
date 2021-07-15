import random

import numpy as np
from game import find_target_index, Game, convert_move
from Basic_Dicts import Pieces, moved


def val_eval(state):
    base_values = {
        0: 0,
        Pieces["Flag"]: 1000,
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

    own_valboard = np.zeros((10, 10))
    op_valboard = np.zeros((10, 10))

    # Counting Material
    # opponents material is hidden normally but we have the same info through the graveyard and op_board, just not
    # the positions
    for x, y in np.ndindex(state[0].shape):
        op_valboard[x][y] = base_values[state[0][x][y]] * (bool(state[1][x][y]))
        own_valboard[x][y] = base_values[state[0][x][y]] * (bool(state[2][x][y]))

    # Adjust spy value if enemy Marshal is dead(de boer invincible)
    reduced_spy = 10
    if Pieces["Marshal"] in state[3][:4]:
        x, y = np.where(own_valboard == Pieces["Spy"])
        if x.size != 0:
            own_valboard[x][y] = reduced_spy
    if Pieces["Marshal"] in state[3][5:]:
        x, y = np.where(op_valboard == Pieces["Spy"])
        if x.size != 0:
            own_valboard[x][y] = reduced_spy

    # Adjust values if pieces are not known to the opponent
    hidden_bonus = 1.3

    grave_pieces = np.count_nonzero(state[3][:6])
    dead_bombs = np.count_nonzero(state[3][:6] == Pieces["Bomb"])
    bombs = 6 - dead_bombs
    alive_pieces = 40 - grave_pieces - bombs - 1
    if alive_pieces <= 15:  # if moving opp pieces are few we want to be aggressive not to hide info
        hidden_bonus = 1.02

    for x, y in np.ndindex(state[0].shape):
        if state[1][x][y] == moved["moved"] or state[1][x][y] == moved["hidden"]:
            op_valboard[x][y] = op_valboard[x][y] * hidden_bonus
        if state[2][x][y] == moved["moved"] or state[2][x][y] == moved["hidden"]:
            own_valboard[x][y] = own_valboard[x][y] * hidden_bonus

    # +++Maybe position with extra value if held(near flag,river)

    result = np.sum(own_valboard) - np.sum(op_valboard) + random.randrange(0, 100) * 0.01
    # result = np.divide(result, 3400)  # bring close to (-1,1)
    return result


class Probabilistic:

    def __init__(self, board: np.array((4, 10, 10))):
        self.state = board

    #  WORKING
    def legal_actions(self):  # legal list
        board = self.state[0]
        opboard = self.state[1]
        ownboard = self.state[2]
        skipped_tiles = 0
        illegal_moves = []

        '''
        while move_index >= 0:
            illegal_moves.append(self.history[move_index])
            move_index -= 2
            if len(illegal_moves) > 3:
                break
        '''
        output = []
        for i in range(10):
            for j in range(10):

                leftex = False
                rightex = False
                forwex = False
                backex = False

                if (j == 2 or j == 3 or j == 6 or j == 7) and (i == 4 or i == 5):
                    skipped_tiles += 1
                    continue
                if ownboard[i][j] == 0:
                    continue
                if board[i][j] == Pieces["Bomb"] or board[i][j] == Pieces["Flag"]:
                    continue

                if (i == 0) or (i == 6 and (j == 2 or j == 3 or j == 6 or j == 7)):
                    forwex = True
                if (i == 9) or (i == 3 and (
                        j == 2 or j == 3 or j == 6 or j == 7)):  # or (i==3 and (j==2 or j==3 or j==6 or j==7))
                    backex = True
                if (j == 0) or ((j == 4 or j == 8) and (i == 5 or i == 4)):
                    leftex = True
                if (j == 9) or ((j == 1 or j == 5) and (i == 4 or i == 5)):  # or ((j==1 or j==5) and (i==4 or i==5 )
                    rightex = True
                # NORMAL MOVE
                if board[i][j] != Pieces["Scout"]:
                    if not leftex:
                        if ownboard[i][j - 1] == 0:
                            check = find_target_index(i, j, i, j - 1)
                            if check not in illegal_moves:
                                output.append(check)

                    if not rightex:
                        if ownboard[i][j + 1] == 0:
                            check = find_target_index(i, j, i, j + 1)
                            if check not in illegal_moves:
                                output.append(check)

                    if not forwex:
                        if ownboard[i - 1][j] == 0:
                            check = find_target_index(i, j, i - 1, j)
                            if check not in illegal_moves:
                                output.append(check)

                    if not backex:
                        if ownboard[i + 1][j] == 0:
                            check = find_target_index(i, j, i + 1, j)
                            if check not in illegal_moves:
                                output.append(check)

                    continue
                # SCOUT MOVE
                if board[i][j] == Pieces["Scout"]:
                    if not leftex:
                        dist = 1
                        maxdist = j
                        if i == 4 or i == 5:
                            maxdist = 1
                        while dist <= maxdist:
                            if ownboard[i][j - dist] == 0:
                                check = find_target_index(i, j, i, j - dist)
                                if check not in illegal_moves:
                                    output.append(check)

                            if opboard[i][j - dist] != 0 or ownboard[i][j - dist] != 0:
                                dist = maxdist + 1
                            dist += 1
                    if not rightex:
                        dist = 1
                        maxdist = 9 - j
                        if i == 4 or i == 5:
                            maxdist = 1
                        while dist <= maxdist:
                            if ownboard[i][j + dist] == 0:
                                check = find_target_index(i, j, i, j + dist)
                                if check not in illegal_moves:
                                    output.append(check)

                            if opboard[i][j + dist] != 0 or ownboard[i][j + dist] != 0:
                                dist = maxdist + 1
                            dist += 1
                    if not forwex:
                        dist = 1
                        maxdist = i
                        if (j == 2 or j == 3 or j == 6 or j == 7) and (i > 6):
                            maxdist = i - 6
                        while dist <= maxdist:
                            if ownboard[i - dist][j] == 0:
                                check = find_target_index(i, j, i - dist, j)
                                if check not in illegal_moves:
                                    output.append(check)

                            if ownboard[i - dist][j] != 0 or opboard[i - dist][j] != 0:
                                dist = maxdist + 1
                            dist += 1
                    if not backex:
                        dist = 1
                        maxdist = 9 - i
                        if (j == 2 or j == 3 or j == 6 or j == 7) and (i < 3):
                            maxdist = 9 - i - 6
                        while dist <= maxdist:
                            if ownboard[i + dist][j] == 0:
                                check = find_target_index(i, j, i + dist, j)
                                if check not in illegal_moves:
                                    output.append(check)

                            if ownboard[i + dist][j] != 0 or opboard[i + dist][j] != 0:
                                dist = maxdist + 1
                            dist += 1

        return output
        # Game specific calculation of legal actions.

    #  DONE
    def hidden_units(self):

        max_pieces = {
            Pieces["Bomb"]: 6,
            Pieces["Marshal"]: 1,
            Pieces["General"]: 1,
            Pieces["Colonel"]: 2,
            Pieces["Major"]: 3,
            Pieces["Captain"]: 4,
            Pieces["Lieutenant"]: 4,
            Pieces["Sergeant"]: 4,
            Pieces["Miner"]: 5,
            Pieces["Scout"]: 8,
            Pieces["Spy"]: 1,
            Pieces["Flag"]: 1
        }
        board = self.state[0]
        opboard = self.state[1]
        grave = self.state[3]
        for x, y in np.ndindex(self.state[0].shape):
            if opboard[x][y] != 0 and opboard[x][y] != moved["moved"] and opboard[x][y] != moved["hidden"]:
                max_pieces[board[x][y]] -= 1
        for x, y in np.ndindex(4, 10):
            if grave[x][y] != 0:
                max_pieces[grave[x][y]] -= 1
        return max_pieces

    #  DONE
    def prepare_state(self, action):
        row, col, trow, tcol = convert_move(action)
        opboard = self.state[1]
        if opboard[trow][tcol] == moved["moved"] or opboard[trow][tcol] == moved["hidden"]:
            temp_state = self.randomize()
        else:
            temp_state = np.copy(self.state)
        return temp_state

    #  DONE
    def randomize(self):
        tstate = np.copy(self.state)
        board = tstate[0]
        opboard = tstate[1]
        units = self.hidden_units()
        list_all = [Pieces["Bomb"], Pieces["Marshal"], Pieces["General"], Pieces["Colonel"], Pieces["Major"],
                    Pieces["Captain"], Pieces["Lieutenant"], Pieces["Sergeant"], Pieces["Miner"], Pieces["Scout"],
                    Pieces["Spy"], Pieces["Flag"]]

        list_movable = [Pieces["Marshal"], Pieces["General"], Pieces["Colonel"], Pieces["Major"],
                        Pieces["Captain"], Pieces["Lieutenant"], Pieces["Sergeant"], Pieces["Miner"], Pieces["Scout"],
                        Pieces["Spy"]]
        for x, y in np.ndindex(10, 10):
            if opboard[x, y] == moved["moved"]:
                pr = [np.divide(units[i], sum(units.values()) - units[Pieces["Bomb"]] - units[Pieces["Flag"]])
                      for i in list_movable]
                piece = np.random.choice(list_movable, size=1, p=pr)[0]

                board[x][y] = piece
                units[piece] -= 1
        for x, y in np.ndindex(10, 10):
            if opboard[x, y] == moved["hidden"]:
                pr = [np.divide(units[i], sum(units.values())) for i in list_all]
                piece = np.random.choice(list_all, size=1, p=pr)[0]

                board[x][y] = piece
                units[piece] -= 1
        # print(tstate)
        # print(units)
        return tstate

    #  TO DO
    @staticmethod
    def evaluate(state):
        return val_eval(state)

    #  SEEMS TO WORK
    @staticmethod
    def apply(tstate, action):  # add action to history (how it changes state?)
        # row,col & target row,col:  row,col,trow,tcol
        # assume the move is legal

        row, col, trow, tcol = convert_move(action)
        '''
        if self.board[row][col] == 0:
            print('ERROR:apply, board = 0')
        if self.ownboard[row][col] == 0:
            print('ERROR:apply, ownboard = 0')
        '''

        board = tstate[0]
        opboard = tstate[1]
        ownboard = tstate[2]
        grave = tstate[3]
        # SIMPLE MOVE
        if board[trow][tcol] == 0:
            board[trow][tcol] = board[row][col]
            board[row][col] = 0
            if ownboard[row][col] == moved["hidden"]:
                ownboard[row][col] = moved["moved"]
            ownboard[trow][tcol] = ownboard[row][col]
            ownboard[row][col] = 0
            if abs(row - trow) > 1 or abs(col - tcol) > 1:
                ownboard[trow][tcol] = board[trow][tcol]

        # SPECIAL ATTACK CASES
        # FLAG CAPTURE
        elif board[trow][tcol] == Pieces["Flag"]:
            # παρολο που το ματς ληγει υπολογιζω το τελικο ποσιτιον
            send_to_graveyard(grave, board[trow][tcol], own=False)
            board[trow][tcol] = board[row][col]
            board[row][col] = 0
            if ownboard[row][col] == moved["hidden"]:
                ownboard[row][col] = moved["moved"]
            ownboard[trow][tcol] = ownboard[row][col]
            ownboard[row][col] = 0
            opboard[trow][tcol] = 0
            # REVEAL SCOUT IF MOVED MULTIPLE TILES:
            if abs(row - trow) > 1 or abs(col - tcol) > 1:
                ownboard[trow][tcol] = board[trow][tcol]

        elif board[trow][tcol] == Pieces["Bomb"]:
            if board[row][col] == Pieces["Miner"]:
                send_to_graveyard(grave, board[trow][tcol], own=False)
                board[trow][tcol] = board[row][col]
                board[row][col] = 0
                ownboard[trow][tcol] = board[trow][tcol]
                ownboard[row][col] = 0
                opboard[trow][tcol] = 0

            else:
                send_to_graveyard(grave, board[row][col], own=True)
                board[row][col] = 0
                ownboard[row][col] = 0
                opboard[trow][tcol] = board[trow][tcol]

        # NORMAL ATTACK
        elif board[row][col] > board[trow][tcol]:
            send_to_graveyard(grave, board[trow][tcol], own=False)
            ownboard[trow][tcol] = board[row][col]
            ownboard[row][col] = 0
            opboard[trow][tcol] = 0
            board[trow][tcol] = board[row][col]
            board[row][col] = 0
        elif board[row][col] == board[trow][tcol]:
            send_to_graveyard(grave, board[trow][tcol], own=False)
            send_to_graveyard(grave, board[row][col], own=True)
            board[row][col] = 0
            board[trow][tcol] = 0
            ownboard[row][col] = 0
            opboard[trow][tcol] = 0
        else:
            # # special cases
            if board[trow][tcol] == Pieces["Marshal"] and board[row][col] == Pieces["Spy"]:
                send_to_graveyard(grave, board[trow][tcol], own=False)
                ownboard[trow][tcol] = board[row][col]
                ownboard[row][col] = 0
                opboard[trow][tcol] = 0
                board[trow][tcol] = board[row][col]
                board[row][col] = 0
            else:
                send_to_graveyard(grave, board[row][col], own=True)
                board[row][col] = 0
                ownboard[row][col] = 0
                opboard[trow][tcol] = board[trow][tcol]


        return

    #  DONE
    def evaluate_moves(self):
        actions = self.legal_actions()
        possible_moves = {}
        for action in actions:
            temp = self.prepare_state(action)
            self.apply(tstate=temp, action=action)
            print(action, "\n", temp)
            value = self.evaluate(temp)
            possible_moves[action] = value
        print(possible_moves)
        return possible_moves


def send_to_graveyard(graveyard, piece, own=False):
    i = 0
    j = 0
    if own:
        i = 6
    while True:
        if graveyard[i][j] == 0:
            graveyard[i][j] = piece
            break
        j += 1
        if j == 10:
            i += 1
            j = 0
    return

