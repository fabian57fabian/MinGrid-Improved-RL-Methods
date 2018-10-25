import os
import shutil
import numpy as np
import json
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--acc", type=float, default=0.9,
                    help="accurancy wanted for final random train (default: 0.9)")
parser.add_argument("--env-size", default="8",
                    help="size of the model (default: 8 as 8x8)")
args = parser.parse_args()

env ="MiniGrid-DoorKey-"+args.env_size+"x"+args.env_size+"-v0"
model="DoorKey-"+args.env_size+"x"+args.env_size+"-stratified"

# Deltas used for training without last one (delta=1 means random over all wall locations)
deltas = np.array([.1,.25,.4,.55,.7,.85,.92])

def train(procs, delta_strat, ending_acc=1, N=5):
    os.system("python3 -m scripts.train --procs "+str(procs)+" --strat "+str(delta_strat)+" --algo=ppo --env " + env + " --no-instr --tb --frames=100000000 --model "+model+" --save-interval 10 --ending-acc "+str(ending_acc))

def main():
    # Starting training frames 
    train(20,deltas[0], ending_acc=0.9)
    iter_deltas = iter(deltas)
    next(iter_deltas)
    for _delta in iter_deltas:
        train(20,_delta , ending_acc=0.9, N=20)
    # The last training with random doors
    train(20,1,ending_acc=args.acc, N=20)

main()
