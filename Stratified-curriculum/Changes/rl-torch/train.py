#!/usr/bin/env python3
# Updated by Fabian: added for copy and mkdir
import os
import shutil
import json
import argparse
import gym
import time
import datetime
import torch
import torch_rl
import sys
import numpy as np

try:
    import gym_minigrid
except ImportError:
    pass

import utils
from model import ACModel

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: ENV_ALGO_TIME)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--ending-acc", type=float, default=1,
                    help="train until reaching a mean of this value as rreturn (default: 1, meaning training stops when reaching accurancy mean=1)")
parser.add_argument("--ending-acc-window", type=int, default=5,
                    help="number of log intervals to check mean for ending_acc (default: 5)")
parser.add_argument("--frames", type=int, default=10 ** 7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate for optimizers (default: 7e-4)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of steps the gradient is propagated back in time (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--no-instr", action="store_true", default=False,
                    help="don't use instructions in the model")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
parser.add_argument("--strat", type=float, default=1,
                    help="delta for stratified curriculum (default 1 meaning no stratification, just random)")
parser.add_argument("--sigma", type=float, default=0.6,
                    help="sigma value for gaussian stratified (default: 0.6)")
parser.add_argument("--save-frames", type=int, default=10000000,
                    help="frames after then save the model into a folder (default 1e7)")
parser.add_argument("--max-steps", type=int, default=-1,
                    help="max steps for DoorKey env (default: -1 as default 10*size*size)")
parser.add_argument("--use-min", action="store_true", default=False,
                    help="use min instead of mean for accurancy")
parser.add_argument("--reward-multiplier", type=float, default=0.9,
                    help="reward multiplier for reward formulae (1-rm * (steps/max_step)). default: 0.9. Lower it is, higher is the reward")
parser.add_argument("--strat-method", default='gicar',
                    help="name of the method to use [gigar, gicar, gidb, gib](default: gicar)")
args = parser.parse_args()

# Define run dir

suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_{}_seed{}_{}".format(args.env, args.algo, args.seed, suffix)
model_name = args.model or default_model_name
save_dir = utils.get_save_dir(model_name)

# Define logger, CSV writer and Tensorboard writer

logger = utils.get_logger(save_dir)
csv_file, csv_writer = utils.get_csv_writer(save_dir)
if args.tb:
    from tensorboardX import SummaryWriter

    tb_writer = SummaryWriter(save_dir)

# Log command and all script arguments

logger.info("{}\n".format(" ".join(sys.argv)))
logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environments

envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(args.seed + 10000 * i, delta_strat=args.strat, gaussian_sigma=args.sigma, strat_method=args.strat_method)
    env.set_reward_multiplier(args.reward_multiplier)
    if args.max_steps != -1:
        # default max steps are 10*size*size, for size=16 is 2560
        env.set_max_steps(args.max_steps)
    envs.append(env)

# Define obss preprocessor

preprocess_obss = utils.ObssPreprocessor(save_dir, envs[0].observation_space)

# Load training status

try:
    status = utils.load_status(save_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}

# Define actor-critic model

try:
    acmodel = utils.load_model(save_dir)
    logger.info("Model successfully loaded\n")
except OSError:
    acmodel = ACModel(preprocess_obss.obs_space, envs[0].action_space, not args.no_instr, not args.no_mem)
    logger.info("Model successfully created\n")
logger.info("{}\n".format(acmodel))

if torch.cuda.is_available():
    acmodel.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Define actor-critic algo

if args.algo == "a2c":
    algo = torch_rl.A2CAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss)
elif args.algo == "ppo":
    algo = torch_rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# Train model

num_frames = status["num_frames"]
total_start_time = time.time()
update = status["update"]

# Updated by FABIAN: generate mean over a window of N log interval
# We use a circular list
mean_acc_array = np.zeros(args.ending_acc_window)
mean_acc_pos = 0
mean_acc_mean = 0


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_agent(src, dest):
    mkdir('storage/' + save_folder + '/' + dest)
    _src = 'storage/' + src + '/model.pt'
    _dest = 'storage/' + save_folder + '/' + dest + '/model.pt'
    shutil.copy2(_src, _dest)
    with open('storage/' + save_folder + '/' + dest + '/status.json', 'w') as outfile:
        json.dump({"num_frames": num_frames, "update": update, "strat": args.strat, "sigma": args.sigma}, outfile)
    print("model successfully saved at frames " + str(num_frames))


# Updated by FABIAN: save model after each args.save_frames frames
last_frame_block = int(num_frames / args.save_frames)
save_folder = "copies_of_" + args.model

if num_frames > args.save_frames:
    copy_agent(args.model, "first-strat-" + str(args.strat) + "-frames-" + str(num_frames))

mkdir("storage/" + save_folder)

while num_frames < args.frames and mean_acc_mean < args.ending_acc:
    # Update model parameters

    update_start_time = time.time()
    logs = algo.update_parameters()
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        # Updated by FABIAN
        # Using this circular list, we remove (array[head] / N) from mean and add (rreturn_mean / N) to mean. then save rreturn_mean and move the tail (being mean_acc_pos)
        accuracy = rreturn_per_episode['min' if args.use_min else 'mean']
        mean_acc_mean = mean_acc_mean - (
                mean_acc_array[(mean_acc_pos + 1) % args.ending_acc_window] / args.ending_acc_window) + (
                                accuracy / args.ending_acc_window)
        mean_acc_pos = (mean_acc_pos + 1) % args.ending_acc_window
        mean_acc_array[mean_acc_pos] = accuracy

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        # data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
        """ old logger, too much data
        logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))
        """

        data += [logs["matches_played"]]
        header += ["Games"]
        logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | Games {}"
                .format(*data))
        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_writer.writerow(header)
        csv_writer.writerow(data)
        csv_file.flush()

        if args.tb:
            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        status = {"num_frames": num_frames, "update": update}
        utils.save_status(status, save_dir)

    # Save vocabulary and model

    if args.save_interval > 0 and update % args.save_interval == 0:
        preprocess_obss.vocab.save()

        if torch.cuda.is_available():
            acmodel.cpu()
        utils.save_model(acmodel, save_dir)
        logger.info("Model successfully saved")
        if torch.cuda.is_available():
            acmodel.cuda()

    # Updated by FABIAN: save a model each save_frames frames
    if int(num_frames / args.save_frames) > last_frame_block:
        last_frame_block = int(num_frames / args.save_frames)
        copy_agent(args.model, "frames-" + str(num_frames))
