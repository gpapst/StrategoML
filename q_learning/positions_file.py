import csv
import os
import queue
import random
import sys
import threading
import time
import pickle
import numpy as np
import multiprocessing as mp
import timeit
import xml.etree.ElementTree as ET
import pandas as pd

path = "Path to xml data"

"""
geting starting positions from xml files
"""


def harvest_pos(filepath):
    """
    Get starting pos from xml file and save to a csv

    :param filepath: path
    """
    csv_path = "G:\\Diploma\\positions.csv"
    pcs = {
        'A': 0., 'B': -0.5, 'C': 0.1, 'D': 0.2, 'E': 0.3, 'F': 0.4, 'G': 0.5, 'H': 0.6, 'I': 0.7, 'J': 0.8, 'K': 0.9,
        'L': 1., 'M': -0.1, 'N': -0.5, 'O': 0.1, 'P': 0.2, 'Q': 0.3, 'R': 0.4, 'S': 0.5, 'T': 0.6, 'U': 0.7, 'V': 0.8,
        'W': 0.9, 'X': 1., 'Y': -0.1, '_': 0.

    }
    column = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9

              }

    tree = ET.parse(filepath)
    root = tree.getroot()
    if root.find(".//field") is None:
        return
    pos = root.find(".//field").attrib['content']
    with open(csv_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([pos])


def check():
    csv_path = "positions.csv"

    d = pd.read_csv(csv_path, header=None, names=["position"])

    print(d['position'][:20])
    print(len(d['position']))


def fill_pos():
    for entry in os.listdir(path):
        print("Starting processing folder:", os.path.join(path, entry))
        for file in os.listdir(os.path.join(path, entry)):
            desc, rest = file.split('-', 1)
            if desc == 'classic':
                fullpath = os.path.join(path, entry)
                harvest_pos(os.path.join(fullpath, file))



