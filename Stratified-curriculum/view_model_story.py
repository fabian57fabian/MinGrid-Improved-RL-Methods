import argparse
import os
import gym
from os import listdir
from scripts.evaluate_all_walls import start
from scripts.evaluate_all_walls import save_all
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from evaluate_model_story import create_plot_3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="name of the model (REQUIRED)")
    args = parser.parse_args()
    test_folder = "storage/copies_of_" + args.model + "/tests"
    assert os.path.isdir(test_folder), "File [" + test_folder + "] don't exist!"

    all_frames_data = []
    filenames = find_filenames(test_folder, '.csv')
    sorted_files = [int(_model[7:-4]) for _model in filenames]
    sorted_files.sort()
    frames = np.zeros(len(sorted_files))
    for i, frame_number in enumerate(sorted_files):
        filename = test_folder + "/frames-" + str(frame_number) + ".csv"
        assert os.path.isfile(filename), "[" + filename + "] doesn't exist!"
        frames[i] = frame_number
        all_frames = np.loadtxt(filename, delimiter=',', unpack=True)
        all_frames_data.append(np.transpose(all_frames))

    _plot = create_plot_3d(all_frames_data, frames)
    _plot.show()


def find_filenames(path_to_dir, suffix):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


if __name__ == "__main__":
    main()
