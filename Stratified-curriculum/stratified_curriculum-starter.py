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

parser = argparse.ArgumentParser()
parser.add_argument("--acc", type=float, default=0.9,
                    help="accurancy wanted for final random train (default: 0.9)")
parser.add_argument("--env-size", type=int, default=16,
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
parser.add_argument("--strat-distance", type=float, default=0.03,
                    help="difference between testing strat (default 0.03 as 0.0, 0.03, 0.06, 0.09, 0.12, 0.15....)")
parser.add_argument("--name", default="test",
                    help="name for this model (default: test)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--use-min", action="store_true", default=False,
                    help="use min instead of mean for accurancy")
parser.add_argument("--use-noise-walls", action="store_true", default=False,
                    help="use noise walls (default false)")
parser.add_argument("--start-from-zero", action="store_true", default=False,
                    help="use a trained model in strat=0.0")
parser.add_argument("--starting-model", default="master_yi",
                    help="name of the model to begin with [master_yi with gicar, garen with gidb] (default: master_yi)")
parser.add_argument("--max-steps", type=int, default=0,
                    help="max steps for DoorKey env (default: 10 * size * size)")
parser.add_argument("--reward-multiplier", type=float, default=0.9,
                    help="reward multiplier for reward formulae (1-rm * (steps/max_step)). default: 0.9. Lower it is, higher is the reward")
parser.add_argument("--strat-method", default='gicar',
                    help="name of the method to use [gigar, gicar, gidb, gib](default: gicar)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
args = parser.parse_args()

if args.max_steps == 0:
    args.max_steps = 10 * args.env_size * args.env_size

env = "MiniGrid-DoorKey-" + str(args.env_size) + "x" + str(args.env_size) + "-v0"
model = args.name

trained_model_path = "storage/" + args.starting_model + "-" + str(args.env_size)

if not os.path.exists(trained_model_path):
    print("BASIC Model in " + trained_model_path + " doesn't exists. Please create it and restart.")
    raise Exception("Missing basic trained model")

# Deltas used for training without last one (delta=1 means random over all wall locations)
deltas = np.arange(args.strat, 1, args.strat_distance)

save_frames = 1000000


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if not os.path.exists("storage/" + model) and not args.start_from_zero:
    src = "storage" + "/DK-" + str(args.env_size) + "-strat"
    dest = "storage/" + model
    print("Creating model based on BASIC model")
    mkdir(dest)
    files = [f for f in listdir(src) if isfile(join(src, f))]
    for _file in files:
        shutil.copyfile(src + '/' + _file, dest + '/' + _file)


def train(procs, delta_strat, N=5):
    os.system("python3 -m scripts.train --procs " + str(procs) + " --strat " + str(delta_strat) + " --sigma " + str(
        args.sigma) + " --algo=ppo --env " + env + " --no-instr --tb --frames=" + str(
        args.frames) + " --model " + model + " --save-interval 10 --ending-acc " + str(
        args.acc) + " --ending-acc-window " + str(N) + " --save-frames " + str(save_frames) + " --discount " + str(
        args.discount) + (" --use-min" if args.use_min else "") + " --reward-multiplier " + str(
        args.reward_multiplier) + " --max-steps " + str(args.max_steps) + " --strat-method " + str(
        args.strat_method) + (" --use-noise-walls" if args.use_noise_walls else "") + " --optim-eps " + str(
        args.optim_eps))
    # Save train status
    with open('storage/' + model + '/info.json', 'w') as outfile:
        json.dump(
            {"delta:": delta_strat, "acc": args.acc, "env-size": args.env_size, "max-frames": args.frames, "N": args.N,
             "procs": args.procs, "name": args.name, "discount": args.discount, "use-min": args.use_min,
             "reward-multiplier": args.reward_multiplier}, outfile)


status_file = "storage/" + model + "/status.json"


def read_last_frames():
    if os.path.exists(status_file):
        with open(status_file) as f:
            data = json.load(f)
            return data["num_frames"]
    else:
        return -1


def main():
    old_frames = -2
    for _delta in deltas:
        new_frames = read_last_frames()
        if new_frames == old_frames:
            print("Closing because max fames limit of " + str(args.frames) + " reached")
            return 0
        else:
            old_frames = new_frames
        train(args.procs, _delta, N=args.N)
    # The last training with random doors
    train(args.procs, 1, N=args.N)


try:
    main()
except KeyboardInterrupt:
    sys.exit()
