import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-size", type=int, default=5,
                    help="size of MiniGrid (default 5 as 5x5)")
args = parser.parse_args()

low = 2
high = args.size-2
plot=False
np.random.seed(1336)


def tests():
    print("Generating numbers in ["+str(low)+", "+str(high)+")")
    # For 10x10, [2, 8[
    for delta in np.arange(0,1,.01):
        x=np.zeros(high+2)
        for j in range(100):
            x[_gaussian_int(low, high, delta)]+=1
        print(str(x) + " delta: "+str(delta))
        print()
        if plot:
            plt.hist(x)
            plt.title("Delta: " + str(delta))
            plt.show()

def _gaussian_int(low, high, delta_strat):
    # Generate 'gaussian' integer in [low,high[
    if low == high-1 : return low
    if delta_strat <= 0.5:
        # Generate random int with gaussian function using self.delta_strat
        rnd = np.random.randint(0,100)
        if rnd > 90 :
            posID = 1
        else:
            posID = 0
        # Starting point + delta_start mapped between low and high + position given by ""X^2"" distribution
        pos = low + int((2*(delta_strat-.02)) * (high - 1 - low)) + posID
        if pos > high:
            pos = high
        return pos
    else:
        # Generate random int in [x, high[, having x setted by delta_strat
        x = 2 * (1 - delta_strat)
        x = low + int((high-1-low)*x)
        return np.random.randint(x, high)

tests()
