import numpy as np
import gym
from gym.utils import seeding


class Stratified:

    def __init__(self, seed, size=5):
        self.np_random_gauss, _ = seeding.np_random(seed + 20)
        self.delta_strat = 1
        self.gaussian_sigma = .4
        self.low = 2
        self.high = size - 2

    def set_strat(self, strat):
        self.delta_strat = strat

    def set_gaussian_sigma(self, sigma):
        self.gaussian_sigma = sigma

    def set_size(self, env_size):
        self.low = 2
        self.high = env_size - 2

    # No sigma control
    def _gaussian_int_direct_big(self):
        # If number is one, then give back random uniform values in [low, high[
        if self.delta_strat == 1: return int(self.np_random_gauss.uniform(self.low, self.high))
        # Generate 'gaussian' integer in [low,high[ getting bigget
        _mean = self.low + ((self.delta_strat / 2) * (self.high - self.low))
        _sigma = self.delta_strat * (self.high - self.low) / 3
        pos = int(self.np_random_gauss.normal(_mean, _sigma))
        pos = self.high - 1 if pos >= self.high else pos
        pos = self.low if pos < self.low else pos
        return pos

    # Sigma-CONTROLLED
    def _gaussian_int_direct(self):
        # If number is one, then give back random uniform values in [low, high[
        if self.delta_strat == 1: return int(self.np_random_gauss.uniform(self.low, self.high))
        # Generate 'gaussian' integer in [low,high[
        _mean = self.low + (self.delta_strat * (self.high - 1 - self.low))
        pos = int(self.np_random_gauss.normal(_mean, self.gaussian_sigma))
        pos = self.high - 1 if pos >= self.high else pos
        pos = self.low if pos < self.low else pos
        return pos

    # Sigma-CONTROLLED
    def _gaussian_int_gaussian_and_random(self):
        # Generate 'gaussian' integer in [low,high[
        if self.low == self.high - 1 or self.delta_strat == 0:
            return self.low
        if self.delta_strat <= 0.5:
            _mean = self.low + ((2 * self.delta_strat) * (self.high - 1 - self.low))
            pos = int(self.np_random_gauss.normal(_mean, self.gaussian_sigma))
            pos = self.high - 1 if pos >= self.high else pos
            pos = self.low if pos < self.low else pos
            return pos
        else:
            # Generate random int in [x, high[, having x setted by delta_strat
            x = 2 * (1 - self.delta_strat)
            x = self.low + int((self.high - 1 - self.low) * x)
            return self.np_random_gauss.randint(x, self.high)

    # No sigma control
    def _gaussian_int_chi_and_random(self):
        # Generate 'gaussian' integer in [low,high[
        if self.low == self.high - 1 or self.delta_strat < .03: return self.low
        if self.delta_strat <= 0.5:
            # Generate random int with gaussian function using self.delta_strat
            base_pos = self.low + int((2 * (self.delta_strat - .02)) * (self.high - 1 - self.low))
            rnd = self.np_random_gauss.randint(0, 100)
            if rnd < 90:
                posID = 0
            else:
                posID = 1
            if rnd < 6 and base_pos > self.low:
                posID = -1
            # Starting point + delta_start mapped between low and high + position given by ""X^2"" distribution
            pos = base_pos + posID
            if pos > self.high:
                pos = self.high
            return pos
        else:
            # Generate random int in [x, high[, having x setted by delta_strat
            x = 2 * (1 - self.delta_strat)
            x = self.low + int((self.high - 1 - self.low) * x)
            return self.np_random_gauss.randint(x, self.high)

    # Stratified function called
    def strat_method_from(self, strat_method):
        if strat_method == 'gidb':
            return self._gaussian_int_direct_big
        elif strat_method == 'gib':
            return self._gaussian_int_direct
        elif strat_method == 'gicar':
            return self._gaussian_int_chi_and_random
        elif strat_method == 'gigar':
            return self._gaussian_int_gaussian_and_random
        else:
            raise AssertionError('Method ' + str(strat_method) + ' not recognized')
