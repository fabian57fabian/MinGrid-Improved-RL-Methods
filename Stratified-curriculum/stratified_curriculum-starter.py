import os
import sys
import shutil
import numpy as np
import json
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--acc", type=float, default=0.9,
                    help="accurancy wanted for final random train (default: 0.9)")
parser.add_argument("--env-size", default="16",
                    help="size of the model (default: 16 as 16x16)")
parser.add_argument("--sigma", type=float, default=0.6,
                    help="sigma for gaussian (default: 0.4)")
parser.add_argument("--strat", type=float, default=0.0,
                    help="strating strat for stratified (default: 0.0)")
parser.add_argument("--frames", type=int, default=15000000,
                    help="max frames to calculate (default: 40000000)")
parser.add_argument("--N", type=int, default=100,
                    help="mean window length (default: 100)")
parser.add_argument("--procs", type=int, default=40,
                    help="procs to start (default: 40)")
parser.add_argument("--name", default="test",
                    help="name for this model (default: test)")
args = parser.parse_args()

env = "MiniGrid-DoorKey-" + args.env_size + "x" + args.env_size + "-v0"
model = "DK-" + args.env_size + "x" + args.env_size + "-N-" + str(args.N) + "-strat-" + args.name

# Deltas used for training without last one (delta=1 means random over all wall locations)
deltas = np.arange(args.strat, 1, .03)

save_frames = 1000000


def train(procs, delta_strat, ending_acc=1.0, N=5):
    os.system("python3 -m scripts.train --procs " + str(procs) + " --strat " + str(delta_strat) + " --sigma " + str(
        args.sigma) + " --algo=ppo --env " + env + " --no-instr --tb --frames=" + str(
        args.frames) + " --model " + model + " --save-interval 10 --ending-acc " + str(
        ending_acc) + " --ending-acc-window " + str(N) + " --save-frames " + str(save_frames))
    # Save train status
    with open('storage/' + model + '/status_stratified.json', 'w') as outfile:
        json.dump({"delta:": delta_strat}, outfile)


def main():
    for _delta in deltas:
        train(args.procs, _delta, ending_acc=0.9, N=args.N)
    # The last training with random doors
    train(args.procs, 1, ending_acc=args.acc, N=args.N)


try:
    main()
except KeyboardInterrupt:
    sys.exit()
