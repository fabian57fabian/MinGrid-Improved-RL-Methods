#!/usr/bin/env python3

import argparse
import gym
import time

try:
    import gym_minigrid
except ImportError:
    pass

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--pause", type=float, default=0.001,
                    help="pause duration between two consequent actions of the agent (the speed, default 0.1)")
parser.add_argument("--wall-fixed", type=int, default=0,
                    help="set only a wall")
args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)

assert args.wall_fixed == 0 or args.wall_fixed > 1 and args.wall_fixed < 16 - 2, "Fixed wall not in [2,13] or not zero"

# Generate environment
env = gym.make(args.env)

# Define agent

save_dir = utils.get_save_dir(args.model)
agent = utils.Agent(save_dir, env.observation_space, args.argmax)

# Run the agent

done = True
wall_id = 1
while True:
    if done:
        wall_id += 1
        if wall_id > 13:
            wall_id = 2
        env.seed(args.seed, fixed_wall_id=(wall_id if args.wall_fixed == 0 else args.wall_fixed))
        obs = env.reset()
        print("Instr:", obs["mission"])

    time.sleep(args.pause)
    renderer = env.render("human")

    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)

    if renderer.window is None:
        break
