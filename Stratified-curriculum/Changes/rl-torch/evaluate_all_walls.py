#!/usr/bin/env python3

import argparse
import gym
import time
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_rl.utils.penv import ParallelEnv

try:
    import gym_minigrid
except ImportError:
    pass

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=16,
                    help="size of the minigrid doorkey (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--save-folder", default="data_test",
                    help="folder name to save inside storage/ (default data_test)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes of evaluation per strat (default: 1000)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--name", default="data_saved",
                    help="name to save (default: data_saved)")
args = parser.parse_args()

# Set seed for all randomness sources

assert os.path.exists("storage/" + args.model), "storage/" + args.model + " doesn't exist"

env_name = "MiniGrid-DoorKey-" + str(args.size) + "x" + str(args.size) + "-v0"

utils.seed(args.seed)
procs = 10
argmax = False

all_data = np.zeros(shape=(args.size, 8))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


main_time = time.time()
for _wall in range(2, args.size - 2):

    # Generate environment
    envs = []

    for i in range(procs):
        env = gym.make(env_name)
        env.seed(args.seed + 10000 * i, fixed_wall_id=_wall)
        envs.append(env)
    env = ParallelEnv(envs)

    # Define agent

    save_dir = utils.get_save_dir(args.model)
    agent = utils.Agent(save_dir, env.observation_space, argmax, procs)
    # print("CUDA available: {}\n".format(torch.cuda.is_available()))

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run the agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(procs, device=agent.device)
    log_episode_num_frames = torch.zeros(procs, device=agent.device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, dones, _ = env.step(actions)
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=agent.device, dtype=torch.float)
        log_episode_num_frames += torch.ones(procs, device=agent.device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=agent.device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("Wall {} | F {} | FPS {:.0f} | D {} | R:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {} {}"
          .format(_wall, num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    all_data[_wall, 0] = return_per_episode["mean"]
    all_data[_wall, 1] = return_per_episode["std"]
    all_data[_wall, 2] = return_per_episode["min"]
    all_data[_wall, 3] = return_per_episode["mean"]

    all_data[_wall, 4] = num_frames_per_episode["mean"]
    all_data[_wall, 5] = num_frames_per_episode["std"]
    all_data[_wall, 6] = num_frames_per_episode["min"]
    all_data[_wall, 7] = num_frames_per_episode["mean"]

main_time = time.time() - main_time
print("Duration: {:.2f}".format(main_time))

save_directory = "storage/" + args.save_folder
mkdir(save_directory)
np.savetxt(save_directory + "/" + args.name + ".csv", all_data, delimiter=",")


def plot_data(walls, data):
    plt.bar(walls, height=data)
    for _w, _h in zip(walls, data):
        plt.text(_w, _h, r'$' + _h + '$')
    plt.xlabel("wall position")
    plt.title("mean accurancy")
    plt.savefig(save_directory + "/accuracy_mean.png")
    plt.show()
