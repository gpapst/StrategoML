from Basic_Dicts import Pieces, moved, max_pieces
import numpy as np
import tkinter
import pandas as pd

POSITIONS = "positions.csv"


class Game(object):

    def __init__(self, history=None, state_history=None, state=None):
        self.q_fixed = None
        self.history = history or []
        self.state_history = state_history or []
        self.image_history = []
        self.two_square = {
            0: [0, [], []],
            1: [0, [], []]}
        self.child_visits = []
        self.state = np.zeros((4, 10, 10))
        self.board = self.state[0]
        self.opboard = self.state[1]
        self.ownboard = self.state[2]
        self.graveyard = self.state[3]
        if state is None:
            self.reset()
        else:
            self.state = state
            self.fix_state()
        self.num_actions = 1656  # 4672 action space size for chess; 11259 for shogi, 362 for Go

    # NEEDS REWORK
    def reset(self):  # TO_DO
        self.state = np.zeros((4, 10, 10))
        self.fix_state()
        self.board[6:] = random_position()
        self.reversepos()
        self.board[6:] = random_position()
        self.opboard[:4] += -1
        self.ownboard[6:] += -1

    def reset_file(self):
        self.history = []
        self.state_history = []
        self.child_visits = []
        temp = initialize_from_file()
        self.state = temp
        self.fix_state()

    # DONE
    def terminal(self):  # Check if game has ended
        # Game specific termination rules.
        if Pieces["Flag"] in self.graveyard:
            return True
        if self.legal_actions() == [] or self.opponent_actions() == []:
            return True
        return False

    # DONE BUT CHECK AGAIN
    def terminal_value(self, to_play):  # to_play= 0,1?
        """
        Game specific value.

        :param to_play:  Must be 0 or 1 (first or second player)
        :return:  0, 1 or -1
        """
        if to_play > 1:
            print(f"WARNING: to_play given is: {to_play}. Converted to {to_play % 2}")
            to_play = to_play % 2
        if not self.terminal():
            return 0
        else:
            out = 0
            for i in range(10):
                for j in range(10):
                    if self.graveyard[i][j] == Pieces["Flag"]:
                        out = 1 if i < 4 else -1
                        break
            if out == 0:
                if not self.legal_actions():
                    out = -1
                elif not self.opponent_actions():
                    out = 1
            if out == 0:
                print("out still 0 but self.terminal is false?")
            if to_play == self.to_play():
                return out
            else:
                return out * -1

    # DONE
    def legal_actions(self):  # legal list

        skipped_tiles = 0
        illegal_moves = []
        move_index = len(self.history) - 2
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
                if self.ownboard[i][j] == 0:
                    continue
                if self.board[i][j] == Pieces["Bomb"] or self.board[i][j] == Pieces["Flag"]:
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
                if self.board[i][j] != Pieces["Scout"]:
                    if not leftex:
                        if self.ownboard[i][j - 1] == 0:
                            check = find_target_index(i, j, i, j - 1)
                            if check not in illegal_moves:
                                output.append(check)

                    if not rightex:
                        if self.ownboard[i][j + 1] == 0:
                            check = find_target_index(i, j, i, j + 1)
                            if check not in illegal_moves:
                                output.append(check)

                    if not forwex:
                        if self.ownboard[i - 1][j] == 0:
                            check = find_target_index(i, j, i - 1, j)
                            if check not in illegal_moves:
                                output.append(check)

                    if not backex:
                        if self.ownboard[i + 1][j] == 0:
                            check = find_target_index(i, j, i + 1, j)
                            if check not in illegal_moves:
                                output.append(check)

                    continue
                # SCOUT MOVE
                if self.board[i][j] == Pieces["Scout"]:
                    if not leftex:
                        dist = 1
                        maxdist = j
                        if i == 4 or i == 5:
                            maxdist = 1
                        while dist <= maxdist:
                            if self.ownboard[i][j - dist] == 0:
                                check = find_target_index(i, j, i, j - dist)
                                if check not in illegal_moves:
                                    output.append(check)

                            if self.opboard[i][j - dist] != 0 or self.ownboard[i][j - dist] != 0:
                                dist = maxdist + 1
                            dist += 1
                    if not rightex:
                        dist = 1
                        maxdist = 9 - j
                        if i == 4 or i == 5:
                            maxdist = 1
                        while dist <= maxdist:
                            if self.ownboard[i][j + dist] == 0:
                                check = find_target_index(i, j, i, j + dist)
                                if check not in illegal_moves:
                                    output.append(check)

                            if self.opboard[i][j + dist] != 0 or self.ownboard[i][j + dist] != 0:
                                dist = maxdist + 1
                            dist += 1
                    if not forwex:
                        dist = 1
                        maxdist = i
                        if (j == 2 or j == 3 or j == 6 or j == 7) and (i > 6):
                            maxdist = i - 6
                        while dist <= maxdist:
                            if self.ownboard[i - dist][j] == 0:
                                check = find_target_index(i, j, i - dist, j)
                                if check not in illegal_moves:
                                    output.append(check)

                            if self.ownboard[i - dist][j] != 0 or self.opboard[i - dist][j] != 0:
                                dist = maxdist + 1
                            dist += 1
                    if not backex:
                        dist = 1
                        maxdist = 9 - i
                        if (j == 2 or j == 3 or j == 6 or j == 7) and (i < 3):
                            maxdist = 9 - i - 6
                        while dist <= maxdist:
                            if self.ownboard[i + dist][j] == 0:
                                check = find_target_index(i, j, i + dist, j)
                                if check not in illegal_moves:
                                    output.append(check)

                            if self.ownboard[i + dist][j] != 0 or self.opboard[i + dist][j] != 0:
                                dist = maxdist + 1
                            dist += 1

        # 2 square rule check
        player = self.to_play()
        sus_moves = []
        if self.two_square[player][0] == 3:
            for x in self.two_square[player][1]:
                r, c = x
                for y in self.two_square[player][2]:
                    tr, tc = y
                    m = find_target_index(r, c, tr, tc)
                    sus_moves.append(m)
        for m in sus_moves:
            if m in output:
                output.remove(m)
        return output
        # Game specific calculation of legal actions.

    # DONE
    def opponent_actions(self):  # legal list

        skipped_tiles = 0
        illegal_moves = []
        move_index = len(self.history) - 1

        while move_index >= 0:
            illegal_moves.append(self.history[move_index])
            move_index -= 2
            if len(illegal_moves) > 3:
                break
        board = reverse(self.board)
        opboard = reverse(self.ownboard)
        ownboard = reverse(self.opboard)
        # grave = self.reverse_grave()
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
                                output.append(find_target_index(i, j, i, j - 1))

                    if not rightex:
                        if ownboard[i][j + 1] == 0:
                            check = find_target_index(i, j, i, j + 1)
                            if check not in illegal_moves:
                                output.append(find_target_index(i, j, i, j + 1))

                    if not forwex:
                        if ownboard[i - 1][j] == 0:
                            check = find_target_index(i, j, i - 1, j)
                            if check not in illegal_moves:
                                output.append(find_target_index(i, j, i - 1, j))

                    if not backex:
                        if ownboard[i + 1][j] == 0:
                            check = find_target_index(i, j, i + 1, j)
                            if check not in illegal_moves:
                                output.append(find_target_index(i, j, i + 1, j))

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
                                    output.append(find_target_index(i, j, i, j - dist))

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
                                    output.append(find_target_index(i, j, i, j + dist))

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
                                    output.append(find_target_index(i, j, i - dist, j))

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
                                    output.append(find_target_index(i, j, i + dist, j))

                            if ownboard[i + dist][j] != 0 or opboard[i + dist][j] != 0:
                                dist = maxdist + 1
                            dist += 1
        reversed_out = []
        for move in output:
            r, c, tr, tc = convert_move(move)
            rm = find_target_index(9 - r, 9 - c, 9 - tr, 9 - tc)
            reversed_out.append(rm)
        return reversed_out
        # return output
        # Game specific calculation of legal actions.

    def clone(self):  # make copy of current game state
        return Game(list(self.history), list(self.state_history), np.copy(self.state))

    # DONE
    def apply(self, action):  # add action to history (how it changes state?)
        # row,col & target row,col:  row,col,trow,tcol
        # assume the move is legal
        self.state_history.append(np.copy(self.state))  # CURRENT STATE ADDED TO HISTORY
        row, col, trow, tcol = convert_move(action)

        '''
        if self.board[row][col] == 0:
            print('ERROR:apply, board = 0')
        if self.ownboard[row][col] == 0:
            print('ERROR:apply, ownboard = 0')
        '''
        # 2 square rule------------
        tp = self.to_play()
        if (row, col) in self.two_square[tp][1] and (trow, tcol) in self.two_square[tp][2]:
            self.two_square[tp][0] += 1
            self.two_square[tp][1] = [(trow, tcol)]
            self.two_square[tp][2].remove((trow, tcol))
            self.two_square[tp][2].append((row, col))
        else:
            if abs(row - trow) > 1:
                assert (col == tcol)
                mx = max(row, trow)
                mn = min(row, trow)
                self.two_square[tp] = [1, [(trow, tcol)], [(x, col) for x in range(mn, mx + 1)]]
                self.two_square[tp][2].remove((trow, tcol))
            if abs(col - tcol) > 1:
                assert (row == trow)
                mx = max(col, tcol)
                mn = min(col, tcol)
                self.two_square[tp] = [1, [(trow, tcol)], [(row, x) for x in range(mn, mx + 1)]]
                self.two_square[tp][2].remove((trow, tcol))
            else:
                self.two_square[tp] = [1, [(trow, tcol)], [(row, col)]]
        # 2 square rule------------
        # SIMPLE MOVE
        if self.board[trow][tcol] == 0:
            self.board[trow][tcol] = self.board[row][col]
            self.board[row][col] = 0
            if self.ownboard[row][col] == moved["hidden"]:
                self.ownboard[row][col] = moved["moved"]
            self.ownboard[trow][tcol] = self.ownboard[row][col]
            self.ownboard[row][col] = 0
            if abs(row - trow) > 1 or abs(col - tcol) > 1:
                self.ownboard[trow][tcol] = self.board[trow][tcol]

        # SPECIAL ATTACK CASES
        # FLAG CAPTURE
        elif self.board[trow][tcol] == Pieces["Flag"]:
            # παρολο που το ματς ληγει υπολογιζω το τελικο ποσιτιον
            self.send_to_graveyard(self.board[trow][tcol], own=False)
            self.board[trow][tcol] = self.board[row][col]
            self.board[row][col] = 0
            if self.ownboard[row][col] == moved["hidden"]:
                self.ownboard[row][col] = moved["moved"]
            self.ownboard[trow][tcol] = self.ownboard[row][col]
            self.ownboard[row][col] = 0
            self.opboard[trow][tcol] = 0
            # REVEAL SCOUT IF MOVED MULTIPLE TILES:
            if abs(row - trow) > 1 or abs(col - tcol) > 1:
                self.ownboard[trow][tcol] = self.board[trow][tcol]

        elif self.board[trow][tcol] == Pieces["Bomb"]:
            if self.board[row][col] == Pieces["Miner"]:
                self.send_to_graveyard(self.board[trow][tcol], own=False)
                self.board[trow][tcol] = self.board[row][col]
                self.board[row][col] = 0
                self.ownboard[trow][tcol] = self.board[trow][tcol]
                self.ownboard[row][col] = 0
                self.opboard[trow][tcol] = 0

            else:
                self.send_to_graveyard(self.board[row][col], own=True)
                self.board[row][col] = 0
                self.ownboard[row][col] = 0
                self.opboard[trow][tcol] = self.board[trow][tcol]

        # NORMAL ATTACK
        elif self.board[row][col] > self.board[trow][tcol]:
            self.send_to_graveyard(self.board[trow][tcol], own=False)
            self.ownboard[trow][tcol] = self.board[row][col]
            self.ownboard[row][col] = 0
            self.opboard[trow][tcol] = 0
            self.board[trow][tcol] = self.board[row][col]
            self.board[row][col] = 0
        elif self.board[row][col] == self.board[trow][tcol]:
            self.send_to_graveyard(self.board[trow][tcol], own=False)
            self.send_to_graveyard(self.board[row][col], own=True)
            self.board[row][col] = 0
            self.board[trow][tcol] = 0
            self.ownboard[row][col] = 0
            self.opboard[trow][tcol] = 0
        else:
            # # special cases
            if self.board[trow][tcol] == Pieces["Marshal"] and self.board[row][col] == Pieces["Spy"]:
                self.send_to_graveyard(self.board[trow][tcol], own=False)
                self.ownboard[trow][tcol] = self.board[row][col]
                self.ownboard[row][col] = 0
                self.opboard[trow][tcol] = 0
                self.board[trow][tcol] = self.board[row][col]
                self.board[row][col] = 0
            else:
                self.send_to_graveyard(self.board[row][col], own=True)
                self.board[row][col] = 0
                self.ownboard[row][col] = 0
                self.opboard[trow][tcol] = self.board[trow][tcol]
        '''
        for i in range(10):
            for j in range(10):
                if self.opboard[i][j] != 0 and self.board[i][j] == 0:
                    print('ERROR1')
                    print(row, col, '->', trow, tcol)
                    print(self.board)
                    print('coordinates:', i, j)
                    print('opboard[][]:', self.opboard[i][j])
                    print('ownboard[][]:', self.ownboard[i][j])
                if self.ownboard[i][j] != 0 and self.board[i][j] == 0:
                    print('ERROR2')
                    print(row, col, '->', trow, tcol)
                    print(self.board)
                    print('coordinates:', i, j)
                if self.board[i][j] != 0 and self.opboard[i][j] == 0 and self.ownboard[i][j] == 0:
                    print('ERROR3')
                    print(row, col, '->', trow, tcol)
                    print(self.board)
                    print('coordinates:', i, j)
        '''
        self.history.append(action)
        self.reversepos()
        return

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())  # sum of child visit_count of root
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)  # self.num_actions = 4672, fixed for game
        ])
        # child_visits[] must have all possible moves because it is the target for the network

        # root.children is a dict, for root.children[0..4762].visit_count calc percentage of time every move is chosen
        # times chosen / total times chosen a move. For the root. and append to the child_visits list of the game
        # so we can do this for every step to fill the list with statistics at every new root(every actual move)

    # DONE
    def make_image(self, state_index: int, formatted=False):  # for the nn?
        """
         Game specific feature planes.

        First is the latest state and then each previous one for (history) moves\n
        Creates the input for the NN training from the position with the given index in history.
        input_planes_used: Number of planes used for each positions information( by extract_image())\n
        history: Number of past moves included in the input
        """
        input_planes_used = 14
        history = 6  # moves included in the image

        if state_index == -1:  # INCLUDE CURRENT STATE
            out = extract_image(self.state)
            current_index = len(self.history)
        else:
            current_index = state_index
            out = extract_image(self.state_history[current_index])

        for i in range(history):
            if current_index - 1 < 0:
                for _ in range(history - i):
                    out = np.concatenate((out, np.zeros((input_planes_used, 10, 10))))
                break
            if np.mod(i, 2) == 0:
                previous_state = reverse_state(self.state_history[current_index - 1])
                previous_state = extract_image(previous_state)
            else:
                previous_state = extract_image(self.state_history[current_index - 1])
            out = np.concatenate((out, previous_state))
            current_index = current_index - 1
        if formatted:
            out = np.rollaxis(out, 0, 3)
        return out

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),  # to play = state_index%2 so 0 or 1 , terminal value: who wins
                self.child_visits[state_index])  # child_visits[s_i] is the choice probs from the mcts

    # state_index is how many moves have been made(len(history) at some point) so to_play can be generated this way
    def make_half_target(self, state_index: int):
        return self.terminal_value(state_index % 2)

    def prepare_images(self):
        if self.terminal():
            self.image_history = []
            for idx in range(len(self.history)):
                img = self.make_image(idx, formatted=True)
                self.image_history.append(img)

        else:
            print("The game has no winner yet")

    def to_play(self):  # 0 or 1
        return len(self.history) % 2

    def display(self):
        display(self.board, self.opboard, self.ownboard)
        return

    def send_to_graveyard(self, piece, own=False):
        i = 0
        j = 0
        if own:
            i = 6
        while True:
            if self.graveyard[i][j] == 0:
                self.graveyard[i][j] = piece
                break
            j += 1
            if j == 10:
                i += 1
                j = 0
        return

    def reverse_grave(self):
        temp = np.zeros((10, 10))
        temp[:4] += self.graveyard[6:]
        temp[6:] += self.graveyard[:4]
        return temp

    def fix_state(self):
        self.board = self.state[0]
        self.opboard = self.state[1]
        self.ownboard = self.state[2]
        self.graveyard = self.state[3]

    # -------------------------------------------------------------------------------------------

    def reversepos(self):
        new = np.zeros((4, 10, 10))

        new[0] = reverse(self.board)
        new[1] = reverse(self.ownboard)
        new[2] = reverse(self.opboard)
        new[3] = self.reverse_grave()
        self.state = new

        self.fix_state()
        return


# HELPERS
def find_target_index(row, col, trow, tcol):
    """
    Finds the position of a move in the nn output (eg move from (4,4) to (0,4) is 756)
    """
    sindex = 0
    index = row * 10 + col
    if index <= 41:
        skip = 0
    elif index <= 45:
        skip = 2
    elif index <= 51:
        skip = 4
    elif index <= 55:
        skip = 6
    else:
        skip = 8
    findex = row * 10 + col - skip
    if row != trow:
        dist = trow - row
        if dist < 0:  # forward
            sindex = row + dist
        if dist > 0:  # back
            sindex = row + dist - 1
    if col != tcol:
        dist = tcol - col
        if dist < 0:  # left
            sindex = col + dist
        if dist > 0:  # right
            sindex = col + dist - 1
        sindex = sindex + 9
    index = findex * 18 + sindex
    return index


def convert_move(action):
    # every 18, change tile, first 9:vertical(row),next 9:horizontal(col)
    if action > 1655:
        print('action:', action, " >1655")
        return 0, 0, 0, 0

    if action < 756:
        skipped = 0
    elif action < 792:
        skipped = 2
    elif action < 864:
        skipped = 4
    elif action < 900:
        skipped = 6
    else:
        skipped = 8
    origin = np.floor_divide(action, 18) + skipped
    target = np.mod(action, 18)
    # print(origin, target)
    row = np.floor_divide(origin, 10)
    # print(row)
    col = origin % 10
    # print(col)
    direction = np.floor_divide(target, 9)
    coordinate = np.mod(target, 9)
    # print(direction, coordinate)

    if direction == 0:
        tcol = col
        if coordinate < row:
            trow = coordinate
        else:
            trow = coordinate + 1
    else:
        trow = row
        if coordinate < col:
            tcol = coordinate
        else:
            tcol = coordinate + 1

    return row, col, trow, tcol


def reverse(board):
    b = np.flip(board, (0, 1))

    return b


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


def extract_image(state):
    """
    Each plane represents:\n
    1.own movable\n
    2.own immovable\n
    3.miners(bin)\n
    4.scouts\n
    5.spy\n
     (from opponents pov)\n
    6.unknown (bin)\n
    7.moved but not revealed(bin)\n
     (opponents board)\n
    8.known movable\n
    9.known bombs(bin)\n
    10.known miners\n
    11.known scouts
    12.unknown(bin)\n
    13.unknown moved(bin)\n
     14.graveyard\n
    :param state: numpy array (4,10,10)
    :return: 14 (10x10) planes
    """
    grave_dict = {
        Pieces["Bomb"]: (1, 0),
        Pieces["Marshal"]: (0, 0),
        Pieces["General"]: (0, 1),
        Pieces["Colonel"]: (0, 2),
        Pieces["Major"]: (0, 3),
        Pieces["Captain"]: (0, 4),
        Pieces["Lieutenant"]: (0, 5),
        Pieces["Sergeant"]: (0, 6),
        Pieces["Miner"]: (0, 7),
        Pieces["Scout"]: (0, 8),
        Pieces["Spy"]: (0, 9),
        Pieces["Flag"]: (1, 1),
    }

    board = np.copy(state[0])
    opboard = np.copy(state[1])
    ownboard = np.copy(state[2])
    grave = np.copy(state[3])
    out = np.zeros((14, 10, 10))
    for i, j in np.ndindex(state[0].shape):
        if board[i][j] != 0:
            if ownboard[i][j] != 0:
                if board[i][j] == Pieces["Bomb"] or board[i][j] == Pieces["Flag"]:
                    out[1][i][j] = board[i][j]
                else:
                    out[0][i][j] = board[i][j]
                    if ownboard[i][j] == moved["hidden"]:
                        out[5][i][j] = 1.
                    elif ownboard[i][j] == moved["moved"]:
                        out[6][i][j] = 1.
                if board[i][j] == Pieces["Miner"]:
                    out[2][i][j] = 1.
                elif board[i][j] == Pieces["Scout"]:
                    out[3][i][j] = 1.
                elif board[i][j] == Pieces["Spy"]:
                    out[4][i][j] = 1.

            elif opboard[i][j] != 0:
                if opboard[i][j] == moved["hidden"]:
                    out[11][i][j] = 1.
                elif opboard[i][j] == moved["moved"]:
                    out[12][i][j] = 1.
                elif opboard[i][j] == Pieces["Bomb"]:
                    out[8][i][j] = board[i][j]
                else:
                    out[7][i][j] = board[i][j]
                    if opboard[i][j] == Pieces["Miner"]:
                        out[9][i][j] = 1.
                    elif opboard[i][j] == Pieces["Scout"]:
                        out[10][i][j] = 1.

    for i, j in np.ndindex((4, 10)):
        if grave[i][j] != 0:
            x, y = grave_dict[grave[i][j]]
            out[13][x][y] += 1

        if grave[i + 6][j] != 0:
            x, y = grave_dict[grave[i + 6][j]]
            out[13][x + 6][y] += 1
    for key, (x, y) in grave_dict.items():
        out[13][x][y] = np.divide(out[13][x][y], (max_pieces[key]))
        out[13][x + 6][y] = np.divide(out[13][x + 6][y], (max_pieces[key]))
    return out


def random_position():
    quantity = {
        Pieces["Flag"]: 1,
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

    }
    position = np.zeros((4, 10))
    step = -1
    indexes = list(range(40))
    for k, v in quantity.items():
        for _ in range(v):
            step += 1
            # print('\nSTEP:', step)
            index = np.random.randint(0, 40 - step)
            index = indexes.pop(index)
            # print(index // 10, index % 10, ":", index)
            # print(indexes)
            position[index // 10][index % 10] = k
    return position


def half(state):
    """
    Reads both positions from file and chooses one to load to the bottom part of the board(state[0]).
    Then reverses the array so that it becomes the top part
    :param state: np.array(x,10,10)
    :return: np.array(x,10,10)
    """
    pcs = {
        'A': 0., 'B': -0.5, 'C': 0.1, 'D': 0.2, 'E': 0.3, 'F': 0.4, 'G': 0.5, 'H': 0.6, 'I': 0.7, 'J': 0.8, 'K': 0.9,
        'L': 1., 'M': -0.1, 'N': -0.5, 'O': 0.1, 'P': 0.2, 'Q': 0.3, 'R': 0.4, 'S': 0.5, 'T': 0.6, 'U': 0.7, 'V': 0.8,
        'W': 0.9, 'X': 1., 'Y': -0.1, '_': 0.

    }
    position = np.random.randint(41500, size=1)[0]

    d = pd.read_csv(POSITIONS, skiprows=position, nrows=1,
                    header=None, names=["position"])
    # print(d['position'][0])

    full_pos = d['position'][0]
    pos, left = full_pos.split('A', 1)
    _, sop = left.rsplit('A', 1)
    choice = np.random.choice(['up', 'down'], 1, p=[0.5, 0.5])[0]

    if choice == 'up':
        temp = np.zeros((10, 10))
        x = 0
        for i, j in np.ndindex(4, 10):
            temp[i][j] = pcs[pos[x]]
            x += 1

        temp = reverse(temp)

        for i, j in np.ndindex(4, 10):
            state[0][i + 6][j] = temp[i + 6][j]
    else:
        x = 0
        for i, j in np.ndindex(4, 10):
            state[0][i + 6][j] = pcs[sop[x]]
            x += 1
    state[0] = reverse(state[0])



def initialize_from_file():
    state = np.zeros((4, 10, 10))
    half(state)
    half(state)
    for x, y in np.ndindex(10, 10):
        if x < 4:
            state[1][x][y] = -1
        elif x > 5:
            state[2][x][y] = -1
    state = mirror_transformation(state, prob=0.5)
    return state


def mirror_transformation(state, prob=0.5):
    choice = np.random.choice([0, 1], 1, p=[prob, 1 - prob])[0]
    if choice == 0:
        # print("op")
        pos1 = np.flip(state[0][:4], axis=1)
        state[0][:4] = pos1
        choice = np.random.choice([0, 1], 1, p=[prob, 1 - prob])[0]
    if choice == 0:
        # print("my")
        pos2 = np.flip(state[0][6:], axis=1)
        state[0][6:] = pos2
    return state


def all_legal_actions(position):  # legal list
    assert np.shape(position) == (4, 10, 10)
    board = position[0]
    opboard = position[1]
    ownboard = position[2]

    skipped_tiles = 0
    illegal_moves = []

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


def display(board, opboard, ownboard):
    window = tkinter.Tk()
    window.title("GUI")
    for x, y in np.ndindex(board.shape):
        if board[x][y] == 0:
            if (x == 4 or x == 5) and (y == 2 or y == 3 or y == 6 or y == 7):
                label = tkinter.Label(window, height=2, width=4, text=".", fg="#027771", bg="#027771",
                                      relief="groove").grid(row=x, column=y)
            else:
                label = tkinter.Label(window, height=2, width=4, text=".", fg="#b3cac3", bg="#b3cac3",
                                      relief="groove").grid(row=x, column=y)
        elif opboard[x][y] != 0:
            if opboard[x][y] == moved["hidden"] or opboard[x][y] == moved["moved"]:
                label = tkinter.Label(window, height=2, width=4, text=str(board[x][y]), fg="#ff844a", bg="#fff0f0",
                                      relief="groove").grid(row=x, column=y)
            else:
                label = tkinter.Label(window, height=2, width=4, text=str(board[x][y]), fg="#940335", bg="#fff0f0",
                                      relief="groove").grid(row=x, column=y)
        elif ownboard[x][y] != 0:
            if ownboard[x][y] == moved["hidden"] or ownboard[x][y] == moved["moved"]:
                label = tkinter.Label(window, height=2, width=4, text=str(board[x][y]), fg="#67d2fe", bg="#f0f0fe",
                                      relief="groove").grid(row=x, column=y)
            else:
                label = tkinter.Label(window, height=2, width=4, text=str(board[x][y]), fg="#000059", bg="#f0f0fe",
                                      relief="groove").grid(row=x, column=y)

    window.mainloop()


'''
#EXAMPLE
g = Game()
g.reset_file()
g.apply(g.legal_actions()[2])
g.display()
g.apply(g.legal_actions()[1])
g.display()
g.apply(g.legal_actions()[0])
g.display()
g.apply(g.legal_actions()[0])
g.display()
'''
