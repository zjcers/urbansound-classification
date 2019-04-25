#!python3

import argparse
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy
import common

def plot_mfcc(mfcc):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    bucket_axis = numpy.arange(0, mfcc.shape[0], 1)
    window_axis = numpy.arange(0, mfcc.shape[1], 1)
    print("Data shape:", mfcc.shape)
    X, Y = numpy.meshgrid(range(mfcc.shape[0]), range(mfcc.shape[1]))
    surf = ax.scatter(X, Y, mfcc)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", dest="file", help="File to process")
    parser.add_argument("--window", dest="window", help="Window length in milliseconds", default=23, type=int)
    parser.add_argument("--buckets", dest="n_mfcc", help="Number of MFCC buckets to divide into", default=13, type=int)
    args = parser.parse_args(sys.argv[1:])
    mfcc = common.process_mfcc(args.file, args.window, args.n_mfcc)
    plot_mfcc(mfcc)