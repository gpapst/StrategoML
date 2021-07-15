import random
import numpy as np
from game import Game, convert_move
from Basic_Dicts import Pieces, moved


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
            if board[x][y] != opboard[x][y]:
                print("sa units error board[x][y] != opboard[x][y]\n", state)
            max_pieces[board[x][y]] -= 1
            if max_pieces[board[x][y]] < 0:
                print("what\n", state)
    for x, y in np.ndindex(4, 10):
        if grave[x][y] != 0:
            max_pieces[grave[x][y]] -= 1
            if max_pieces[grave[x][y]] < 0:
                print("whatG\n", state)
    return max_pieces


def partial_info(state, action: int):
    row, col, trow, tcol = convert_move(action)
    opboard = state[1]
    if opboard[trow][tcol] == moved["moved"]:
        return "moved"
    if opboard[trow][tcol] == moved["hidden"]:
        return "hidden"
    return "none"


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


def randomize(state):
    tstate = np.copy(state)
    board = tstate[0]
    opboard = tstate[1]
    units = hidden_units(tstate)
    # print(tstate)
    # print("INITIAL",units)

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
            # print(x,y,"/nmoved pr:",pr)
            piece = np.random.choice(list_movable, size=1, p=pr)[0]

            board[x][y] = piece
            units[piece] -= 1
            # print("new:",units," ",piece)
            # input("continue: ")
    for x, y in np.ndindex(10, 10):
        if opboard[x, y] == moved["hidden"]:
            pr = [np.divide(units[i], sum(units.values())) for i in list_all]
            # print(x,y,"/n hidden pr:", pr)
            piece = np.random.choice(list_all, size=1, p=pr)[0]

            board[x][y] = piece
            units[piece] -= 1
            # print("new:", units, " ", piece)
            # input("continue: ")
    # print(tstate)
    # print(units)
    return tstate


def evaluate(state):
    base_values = {
        0: 0,
        Pieces["Flag"]: 600,
        Pieces["Marshal"]: 400,
        Pieces["General"]: 300,
        Pieces["Colonel"]: 175,
        Pieces["Major"]: 140,
        Pieces["Captain"]: 100,
        Pieces["Lieutenant"]: 50,
        Pieces["Sergeant"]: 25,
        Pieces["Miner"]: 240,
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

    return result


def search_agent(game: Game):
    list_all = [Pieces["Bomb"], Pieces["Marshal"], Pieces["General"], Pieces["Colonel"], Pieces["Major"],
                Pieces["Captain"], Pieces["Lieutenant"], Pieces["Sergeant"], Pieces["Miner"], Pieces["Scout"],
                Pieces["Spy"], Pieces["Flag"]]

    list_movable = [Pieces["Marshal"], Pieces["General"], Pieces["Colonel"], Pieces["Major"],
                    Pieces["Captain"], Pieces["Lieutenant"], Pieces["Sergeant"], Pieces["Miner"], Pieces["Scout"],
                    Pieces["Spy"]]
    list_unmovable = [Pieces["Bomb"], Pieces["Flag"]]

    possible_moves = {}
    moves = game.legal_actions()
    org_board = np.copy(game.state)

    g_sum = np.sum(org_board[3])
    idle_penalty = 0
    i = 1
    while i <= len(game.state_history):
        test_sum = np.sum(game.state_history[-i][3])
        if test_sum != g_sum:
            break
        i += 1
        idle_penalty += 1

    h = hidden_units(org_board)

    # check if a hidden position can be occupied by a movable piece
    used_list = list_all
    possible = sum(h.values()) - sum(h[x] for x in list_unmovable)
    find_possible = np.count_nonzero(org_board[1] == moved["moved"])
    if possible == find_possible:
        used_list = list_unmovable
    #############
    for move in moves:

        if partial_info(org_board, move) == "moved":
            final_value = 0
            row, col, trow, tcol = convert_move(move)
            for p in list_movable:
                if h[p] > 0:
                    # print("INSTALLING ",p," IN (",trow,",",tcol, ")")
                    temp = np.copy(org_board)
                    temp[0][trow][tcol] = p
                    temp[1][trow][tcol] = p
                    temp = randomize(temp)  # randomize kateyueian
                    apply(temp, action=move)
                    value = evaluate(temp)
                    final_value += value * np.divide(h[p], sum(h.values()) - h[Pieces["Bomb"]] - h[Pieces["Flag"]])
            possible_moves[move] = final_value

        if partial_info(org_board, move) == "hidden":

            final_value = 0
            row, col, trow, tcol = convert_move(move)
            for p in used_list:
                if h[p] > 0:
                    # print("INSTALLING ", p, " IN (", trow, ",", tcol, ")")
                    temp = np.copy(org_board)
                    temp[0][trow][tcol] = p
                    temp[1][trow][tcol] = p
                    temp = randomize(temp)  # randomize kateyueian
                    apply(temp, action=move)
                    value = evaluate(temp)
                    final_value += value * np.divide(h[p], sum(h.values()))

            possible_moves[move] = final_value

        else:
            row, col, trow, tcol = convert_move(move)

            if trow < row:
                penalty = np.divide(idle_penalty, 2)
            else:
                penalty = idle_penalty
            temp = np.copy(org_board)
            apply(temp, action=move)
            value = evaluate(temp) - penalty
            possible_moves[move] = value

    return possible_moves

