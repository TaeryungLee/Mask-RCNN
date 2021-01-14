import os
import sys


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        