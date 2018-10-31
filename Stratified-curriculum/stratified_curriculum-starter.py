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
parser.add_argument("--sigma", type=float, default=0.6,
                    help="sigma for gaussian (default: 0.6)")
parser.add_argument("--strat", type=float, default=0.0,
                    help="strating strat for stratified (default: 0.0)")
parser.add_argument("--frames", type=int, default=15000000,
                    help="max frames to calculate (default: 15000000)")
parser.add_argument("--name", default="test",
                    help="name for this model (default: test)")
args = parser.parse_args()

env ="MiniGrid-DoorKey-"+args.env_size+"x"+args.env_size+"-v0"
model="DoorKey-"+args.env_size+"x"+args.env_size+"-stratified-" + args.name

# Deltas used for training without last one (delta=1 means random over all wall locations)
deltas = np.arange(args.strat,1,.03)


def train(procs, delta_strat, ending_acc=1, N=5):
    os.system("python3 -m scripts.train --procs "+str(procs)+" --strat "+str(delta_strat)+" --sigma "+str(args.sigma)+ " --algo=ppo --env " + env + " --no-instr --tb --frames="+str(args.frames)+" --model "+model+" --save-interval 10 --ending-acc "+str(ending_acc) + " --ending-acc-window "+str(N))
    # Save train status
    with open('storage/'+model+'/status_stratified.json', 'w') as outfile:
            json.dump({"delta:": delta_strat}, outfile)    

def main():
    for _delta in deltas:
        train(20,_delta , ending_acc=0.9, N=100)
    # The last training with random doors
    train(20,1,ending_acc=args.acc, N=50)

main()
