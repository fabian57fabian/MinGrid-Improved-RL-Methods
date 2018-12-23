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
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent")
parser.add_argument("--use-noise-walls", action="store_true", default=False,
                    help="use noise walls (default false)")
parser.add_argument("--strat", type=float, default=1,
                    help="delta for stratified curriculum (default 1 meaning no stratification, just random)")
parser.add_argument("--sigma", type=float, default=0.3,
                    help="sigma value for gaussian stratified (default: 0.6)")
parser.add_argument("--strat-method", default='gicar',
                    help="name of the method to use [gigar, gicar, gidb, gib](default: gicar)")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
strat = delta_strat = args.strat
env.seed(args.seed, delta_strat=strat, gaussian_sigma=args.sigma, strat_method=args.strat_method)
env.set_noise_walls(args.use_noise_walls)

# Define agent

save_dir = utils.get_save_dir(args.model)
agent = utils.Agent(save_dir, env.observation_space, args.argmax)

# Run the agent

done = True

while True:
    if done:
        strat += .02
        if strat > 1:
            strat = 0
        env.seed(args.seed, strat, gaussian_sigma=args.sigma, strat_method=args.strat_method)
        obs = env.reset()
        print("Instr:", obs["mission"])

    time.sleep(args.pause)
    renderer = env.render("human")

    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)

    if renderer.window is None:
        break
