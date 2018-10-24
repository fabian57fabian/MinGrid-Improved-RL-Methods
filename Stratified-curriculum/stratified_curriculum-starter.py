import os
import shutil
import numpy as np
import json
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--frames-per-train", type=int, default=100000,
                    help="number of frames of training (default: 10e5)")
parser.add_argument("--frames-starting", type=int, default=1000000,
                    help="number of frames of training for starting train (default: 1e6)")
parser.add_argument("--frames-rnd-train", type=int, default=500000,
                    help="number of frames of training for last training with random doors (default: 5*1e5)")
parser.add_argument("--env-size", default="8",
                    help="size of the model (default: 8 as 8x8)")
args = parser.parse_args()

env ="MiniGrid-DoorKey-"+args.env_size+"x"+args.env_size+"-v0"
model="DoorKey-"+args.env_size+"x"+args.env_size+"-stratified"

# Deltas used for training without last one (delta=1 means random over all wall locations)
deltas = np.array([.1,.25,.4,.55,.7,.85,.92])

def train(procs, delta_strat, frames):
    os.system("python3 -m scripts.train --procs "+str(procs)+" --strat "+str(delta_strat)+" --algo=ppo --env " + env + " --no-instr --tb --frames="+str(frames)+" --model "+model+" --save-interval 10")

def main():
    # Starting training frames 
    frames = args.frames_starting
    for _delta in deltas:
        train(20,_delta, frames)
        frames += args.frames_per_train
    # The last training with random doors
    frames +=args.frames_rnd_train - args.frames_per_train
    train(20,1,frames)

main()
