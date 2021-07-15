import sys
import tkinter
import numpy as np
from game import Game

sys.path.append("C:\\Users\\user\\PycharmProjects\\Stratego")
from Basic_Dicts import moved, Pieces

dick = {
    -0.1: "F",
    -0.5: "B",
    1: 10,
    0.9: 9,
    0.8: 8,
    0.7: 7,
    0.6: 6,
    0.5: 5,
    0.4: 4,
    0.3: 3,
    0.2: 2,
    0.1: 1,
    0: ""
}


# f0f0fe
# f9d1c7
# fff0f0
def display(board, opboard, ownboard):
    window = tkinter.Tk()
    window.title("GUI")
    for x, y in np.ndindex(board.shape):
        if board[x][y] == 0:
            if (x == 4 or x == 5) and (y == 2 or y == 3 or y == 6 or y == 7):
                label = tkinter.Label(window, height=2, width=4, text=".", fg="#027771", bg="#027771",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
            else:
                label = tkinter.Label(window, height=2, width=4, text=".", fg="#b3cac3", bg="#b3cac3",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
        elif opboard[x][y] != 0:
            if opboard[x][y] == moved["hidden"] or opboard[x][y] == moved["moved"]:
                label = tkinter.Label(window, height=2, width=4, text=str(dick[board[x][y]]), fg="#ff844a",
                                      bg="#f9d1c7",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
            else:
                label = tkinter.Label(window, height=2, width=4, text=str(dick[board[x][y]]), fg="#940335",
                                      bg="#f9d1c7",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
        elif ownboard[x][y] != 0:
            if ownboard[x][y] == moved["hidden"] or ownboard[x][y] == moved["moved"]:
                label = tkinter.Label(window, height=2, width=4, text=str(dick[board[x][y]]), fg="#67d2fe",
                                      bg="#d4e5f9",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
            else:
                label = tkinter.Label(window, height=2, width=4, text=str(dick[board[x][y]]), fg="#000059",
                                      bg="#d4e5f9",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)

    window.mainloop()


def p1display(board, opboard, ownboard):
    window = tkinter.Tk()
    window.title("GUI")
    for x, y in np.ndindex(board.shape):
        if board[x][y] == 0:
            if (x == 4 or x == 5) and (y == 2 or y == 3 or y == 6 or y == 7):
                label = tkinter.Label(window, height=2, width=4, text=".", fg="#027771", bg="#027771",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
            else:
                label = tkinter.Label(window, height=2, width=4, text=".", fg="#b3cac3", bg="#b3cac3",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
        elif opboard[x][y] != 0:
            if opboard[x][y] == moved["hidden"]:
                label = tkinter.Label(window, height=2, width=4, text=".", fg="#f9d1c7",
                                      bg="#f9d1c7",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
            elif opboard[x][y] == moved["moved"]:
                label = tkinter.Label(window, height=2, width=4, text="m", fg="#ff844a",
                                      bg="#f9d1c7",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
            else:
                label = tkinter.Label(window, height=2, width=4, text=str(dick[board[x][y]]), fg="#940335",
                                      bg="#f9d1c7",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
        elif ownboard[x][y] != 0:
            if ownboard[x][y] == moved["hidden"] or ownboard[x][y] == moved["moved"]:
                label = tkinter.Label(window, height=2, width=4, text=str(dick[board[x][y]]), fg="#67d2fe",
                                      bg="#d4e5f9",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)
            else:
                label = tkinter.Label(window, height=2, width=4, text=str(dick[board[x][y]]), fg="#000059",
                                      bg="#d4e5f9",
                                      relief="groove", font="Helvetica 12 bold").grid(row=x, column=y)

    window.mainloop()


"""
# EXAMPLE
g = Game()
g.reset_file()
display(g.state[0], g.state[1], g.state[2])
"""
