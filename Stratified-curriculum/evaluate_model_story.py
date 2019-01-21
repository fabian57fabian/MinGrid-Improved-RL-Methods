import argparse
import os
import gym
from os import listdir

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")

args = parser.parse_args()
path = "storage/copies_of_" + args.model
assert os.path.exists(path) and not path == ""

for model in listdir(path):
    # get fame number from model name
    frames = model[7:]
    os.system(
        "python3 -m scripts.evaluate_all_walls --model copies_of_" + args.model + "/" + model
        + " --frames-at " + frames + " --save-folder copies_of_" + model + "/tests --episodes 20 --name data_of_" + model)
