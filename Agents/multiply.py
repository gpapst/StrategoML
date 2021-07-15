import numpy as np
from game import find_target_index, convert_move
from Basic_Dicts import Pieces, moved


def redecka_eval(state):
    summer = 0
    pw = 1.0
    rw = 0.05
    mw = 0.03
    dw = 0.02
    rank = 0
    for i, j in np.ndindex(10, 10):
        if state[0][i][j] != 0:
            if state[2][i][j] != 0:  # own piece
                summer += pw
                if state[2][i][j] == state[0][i][j]:
                    summer -= rw * rank
                elif state[2][i][j] == moved["moved"]:
                    summer -= mw
                if i > 3:
                    summer -= dw * (i - 3) * (i - 3)
            elif state[1][i][j] != 0:  # opp piece
                summer -= pw
                if state[1][i][j] == state[0][i][j]:  # revealed
                    summer += rw
                elif state[1][i][j] == moved["moved"]:
                    summer += mw
                if i < 6:
                    summer += dw * (6 - i) * (6 - i)
    return summer


class MultiPly:

    def __init__(self, plies=3):
        self.plies = plies

    #  WORKING
    @staticmethod
    def legal_actions(tstate):  # legal list
        board = tstate[0]
        opboard = tstate[1]
        ownboard = tstate[2]
        grave = tstate[3]
        skipped_tiles = 0
        illegal_moves = []
        if Pieces["Flag"] in grave:
            return []

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
    @staticmethod
    def hidden_units(state):

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
        board = state[0]
        opboard = state[1]
        grave = state[3]
        for x, y in np.ndindex(state[0].shape):
            if opboard[x][y] != 0 and opboard[x][y] != moved["moved"] and opboard[x][y] != moved["hidden"]:
                max_pieces[board[x][y]] -= 1
        for x, y in np.ndindex(4, 10):
            if grave[x][y] != 0:
                max_pieces[grave[x][y]] -= 1
        return max_pieces

    #  DONE
    '''
    def prepare_state(self, action):
        row, col, trow, tcol = convert_move(action)
        opboard = self.state[1]
        if opboard[trow][tcol] == moved["moved"] or opboard[trow][tcol] == moved["hidden"]:
            temp_state = self.randomize()
        else:
            temp_state = np.copy(self.state)
        return temp_state

    '''

    #  DONE
    def randomize(self, state):
        tstate = np.copy(state)
        board = tstate[0]
        opboard = tstate[1]
        units = self.hidden_units(state)
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
        return redecka_eval(state)

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
    def evaluate_moves(self, state, plies=2):

        actions = self.legal_actions(state)
        possible_moves = {}
        for action in actions:
            next_state = self.randomize(state)  # RANDOMIZE ENEMY
            self.apply(tstate=next_state, action=action)
            # print(action, "\n", next_state)
            temp_plies = plies - 1
            if temp_plies == 0:
                value = self.evaluate(next_state)
                possible_moves[action] = value

            elif temp_plies > 0:
                next_reversed = reverse_state(next_state)

                opponent_moves = self.evaluate_moves(next_reversed, temp_plies)
                if not bool(opponent_moves):
                    value = 1000
                else:
                    op_move = max(opponent_moves, key=opponent_moves.get)
                    self.apply(next_reversed, op_move)
                    next_reversed_answered = reverse_state(next_reversed)

                    value = self.evaluate(next_reversed_answered)
                possible_moves[action] = value

        # print(possible_moves)
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


def reverse_grave(graveyard):
    temp = np.zeros((10, 10))
    temp[:4] += graveyard[6:]
    temp[6:] += graveyard[:4]
    return temp


def reverse_state(state):
    """
        State from the opponents pov originally: board, my pieces, opponents pieces, grave.
        We want the state from our pov so: board r, opponents pieces r, my pieces r, grave reversed
    """
    out = np.zeros((4, 10, 10))
    out[0] = np.flip(state[0], (0, 1))
    out[1] = np.flip(state[2], (0, 1))
    out[2] = np.flip(state[1], (0, 1))
    out[3] = reverse_grave(state[3])
    return out


def reverse_move(action):
    row, col, trow, tcol = convert_move(action)
    row = 9 - row
    col = 9 - col
    trow = 9 - trow
    tcol = 9 - tcol
    return find_target_index(row, col, trow, tcol)


















