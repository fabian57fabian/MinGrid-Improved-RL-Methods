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

steps = np.array([1000,1500,2560,3000,4000,5000,6000,8000,10000])

def main():
    for step in steps:
        os.system("python stratified_curriculum-starter.py --acc 0.90 --sigma 0.5 --frames 13000000 --N 50 --procs 40 --name train_STEPS_"+str(step)+" --discount 0.99 --reward-multiplier 0.8 --strat-method gicar --strat-distance 0.3 --max-steps "+str(step))

main()
