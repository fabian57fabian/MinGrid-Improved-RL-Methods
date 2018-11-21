import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-size", type=int, default=17,
                    help="size of MiniGrid (default 5 as 5x5)")
parser.add_argument("-sigma", type=float, default=0.55,
                    help="sigma for gaussians (default 0.6)")
parser.add_argument("-type", default="gigar",
                    help="type of test (gigar, gicar, gidb, gid)")
args = parser.parse_args()

# Those are high and low values (e.g. for 10x10 are [2, 8[ )
low = 2
high = args.size - 2
plot = False


# np.random.seed(1336)


def tests():
    print("Generating numbers in [" + str(low) + ", " + str(high) + ")")

    # Generating testing class
    cl_generator = test_method(0, args.sigma)
    generator = None
    generator = cl_generator._gaussian_int_gaussian_and_random if args.type == "gigar" else generator
    generator = cl_generator._gaussian_int_direct_big if args.type == "gidb" else generator
    generator = cl_generator._gaussian_int_direct if args.type == "gid" else generator
    generator = cl_generator._gaussian_int_chi_and_random if args.type == "gicar" else generator

    for delta in np.arange(0, 1.01, .01):
        x = np.zeros(high + 2)
        for j in range(99):
            cl_generator.set_delta(delta)
            x[generator(low, high)] += 1
        print(str(x) + " delta: " + str(delta))
        if plot:
            plt.hist(x)
            plt.title("Delta: " + str(delta))
            plt.show()


class test_method:
    def __init__(self, delta_strat, gaussian_sigma):
        self.delta_strat = delta_strat
        self.gaussian_sigma = gaussian_sigma

    def set_delta(self, delta):
        self.delta_strat = delta

    def set_sigma(self, sigma):
        self.gaussian_sigma = sigma

    class np_random_gauss:
        def uniform(low, high):
            return np.random.uniform(low, high, 1)

        def randint(low, high):
            return np.random.randint(low, high)

        def normal(_mean, _sigma):
            return np.random.normal(_mean, _sigma)

    # No sigma control
    def _gaussian_int_direct_big(self, low, high):
        # If number is one, then give back random uniform values in [low, high[
        if self.delta_strat == 1: return int(self.np_random_gauss.uniform(low, high))
        # Generate 'gaussian' integer in [low,high[ getting bigget
        _mean = low + ((self.delta_strat / 2) * (high - low))
        _sigma = self.delta_strat * (high - low) / 3
        pos = int(self.np_random_gauss.normal(_mean, _sigma))
        pos = high - 1 if pos >= high else pos
        pos = low if pos < low else pos
        return pos

    # Sigma-CONTROLLED
    def _gaussian_int_direct(self, low, high):
        # If number is one, then give back random uniform values in [low, high[
        if self.delta_strat == 1: return int(self.np_random_gauss.uniform(low, high))
        # Generate 'gaussian' integer in [low,high[
        _mean = low + (self.delta_strat * (high - 1 - low))
        pos = int(self.np_random_gauss.normal(_mean, self.gaussian_sigma))
        pos = high - 1 if pos >= high else pos
        pos = low if pos < low else pos
        return pos

    # Sigma-CONTROLLED
    def _gaussian_int_gaussian_and_random(self, low, high):
        # Generate 'gaussian' integer in [low,high[
        if low == high-1 or self.delta_strat == 0: 
            return low
        if self.delta_strat <= 0.5:
            _mean = low + ((2 * self.delta_strat) * (high - 1 - low))
            pos = int(self.np_random_gauss.normal(_mean, self.gaussian_sigma))
            pos = high - 1 if pos >= high else pos
            pos = low if pos < low else pos
            return pos
        else:
            # Generate random int in [x, high[, having x setted by delta_strat
            x = 2 * (1 - self.delta_strat)
            x = low + int((high - 1 - low) * x)
            return self.np_random_gauss.randint(x, high)

    # No sigma control
    def _gaussian_int_chi_and_random(self, low, high):
        # Generate 'gaussian' integer in [low,high[
        if low == high - 1 or self.delta_strat < .03: return low
        if self.delta_strat <= 0.5:
            # Generate random int with gaussian function using self.delta_strat
            base_pos = low + int((2 * (self.delta_strat - .02)) * (high - 1 - low))
            rnd = self.np_random_gauss.randint(0, 100)
            if rnd < 90:
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
            x = low + int((high - 1 - low) * x)
            return self.np_random_gauss.randint(x, high)


tests()
