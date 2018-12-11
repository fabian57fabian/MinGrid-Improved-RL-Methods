import os
import sys
import shutil
import numpy as np
import json
import argparse
import logging
from os import listdir
from os.path import isfile, join
import time

def train(steps):
    os.system("python stratified_curriculum-starter.py --acc 0.90 --sigma 0.5 --frames 13000000 --N 50 --procs 40 --name train_STEPS_"+str(steps)+" --discount 0.99 --reward-multiplier 0.8 --strat-method gicar --strat-distance 0.3 --max-steps "+str(steps))

def main():
    train(1000)
    train(1500)
    train(2560)
    train(3000)
    train(4000)
    train(5000)
    train(6000)
    train(8000)
    train(9000)
    train(10000)

main()
