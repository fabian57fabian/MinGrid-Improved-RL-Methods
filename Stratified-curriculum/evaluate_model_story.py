import argparse
import os
import gym
from os import listdir
from scripts.evaluate_all_walls import start
from scripts.evaluate_all_walls import save_all
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="name of the trained model inside storage folder (REQUIRED)")
    parser.add_argument("--show-graphs", action="store_true", default=False,
                        help="show graph in ending")
    args = parser.parse_args()
    path = "storage/copies_of_" + args.model
    assert os.path.exists(path) and not path == "", "Path [" + path + "] doesn't exist"

    save_floder = "copies_of_" + args.model + "/tests"
    all_frames_data = []

    # take directory list, check if contains model inside, remove initial 'frames-' in name
    # then sort as int
    models = []
    for _copy_of in listdir(path):
        if os.path.isfile("storage/copies_of_" + args.model + "/" + _copy_of + "/model.pt"):
            models.append(_copy_of)
    models = [int(_model[7:]) for _model in models]
    models.sort()

    for _model in models:
        model_with_frames = "frames-" + str(_model)
        model_path = "copies_of_" + args.model + "/" + model_with_frames
        all_data = start(model_path, 0, 20, 16)
        all_frames_data.append(all_data)
        save_all(all_data, save_floder, model_with_frames)

    _plt = create_plot_3d(all_frames_data, models)
    _plt.savefig("storage/" + save_floder + "/accuracy_mean.png")
    if args.show_graphs:
        _plt.show()


def create_plot_3d(data, models):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, all_data in enumerate(data):
        xs = np.arange(0, all_data.shape[0], 1, dtype=int)
        ys = all_data[:, 0]
        ax.bar(xs, ys, zs=models[i], zdir='y', color=colors[i], alpha=0.8)
    ax.set_xlabel('walls')
    ax.set_ylabel('trains')
    ax.set_zlabel('mean accurancy')

    return plt


if __name__ == "__main__":
    main()
