from numpy import *


def img2vector(filename):
    """convert a n×n matrix to a 1×n matrix (vector)"""
    vector = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                vector[0, 32 * i + j] = int(lineStr[j])
    return vector
