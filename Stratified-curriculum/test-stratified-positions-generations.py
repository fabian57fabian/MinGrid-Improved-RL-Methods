import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-size", type=int, default=5,
                    help="size of MiniGrid (default 5 as 5x5)")
parser.add_argument("-sigma", type=float, default=0.6,
                    help="sigma for gaussians (default 0.6)")
args = parser.parse_args()

low = 2
high = args.size-2
plot=False
#np.random.seed(1336)


def tests():
    print("Generating numbers in ["+str(low)+", "+str(high)+")")
    # For 10x10, [2, 8[
    for delta in np.arange(0,1,.01):
        x=np.zeros(high+2)
        for j in range(99):
            meth = test_method(delta, args.sigma)
            x[meth._gaussian_int(low, high)]+=1
        print(str(x) + " delta: "+str(delta))
        if plot:
            plt.hist(x)
            plt.title("Delta: " + str(delta))
            plt.show()

class test_method:
    def __init__(self, delta_strat, gaussian_sigma):
        self.delta_strat=delta_strat
        self.gaussian_sigma = gaussian_sigma

    class np_random_gauss:
        def randint(low, high):
            return np.random.randint(low, high)
        def normal(_mean, _sigma):
            return np.random.normal(_mean,_sigma)

    def _gaussian_int(self, low, high):
        # Generate 'gaussian' integer in [low,high[
        if self.delta_strat <= 0.5:
            _mean = 2 + ((2 * self.delta_strat) * (high - 1 - low))
            pos = int(self.np_random_gauss.normal(_mean, self.gaussian_sigma))
            pos = high-1 if pos >= high else pos
            pos = low if pos < low else pos
            return pos
        else:
            # Generate random int in [x, high[, having x setted by delta_strat
            x = 2 * (1 - self.delta_strat)
            x = low + int((high - 1 - low) * x)
            return self.np_random_gauss.randint(x, high)

    def _gaussian_int_2(self, low, high):
        # Generate 'gaussian' integer in [low,high[
        if low == high-1 or self.delta_strat < .03 : return low
        if self.delta_strat <= 0.5:
            # Generate random int with gaussian function using self.delta_strat
            base_pos = low + int((2*(self.delta_strat-.02)) * (high - 1 - low))
            rnd = self.np_random.randint(0,100)
            if rnd < 90 :
                posID = 0
            else:
                posID = 1
            if rnd < 6 and base_pos > low:
                posID = -1
            # Starting point + delta_start mapped between low and high + position given by ""X^2"" distribution
            pos = base_pos + posID
            if pos > high:
                pos = high
            return pos
        else:
            # Generate random int in [x, high[, having x setted by delta_strat
            x = 2 * (1 - self.delta_strat)
            x = low + int((high-1-low)*x)
            return self.np_random.randint(x, high)


tests()
