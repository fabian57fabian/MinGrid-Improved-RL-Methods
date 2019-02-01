import argparse
import os
from os import listdir
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="name of the model (REQUIRED)")
    parser.add_argument("--use-3d", action="store_true", default=False,
                        help="Ue 3d model instead of different histograms list")
    parser.add_argument("--type-of-result", type=int, default=0,
                        help="rreturn: [0: mean, 1: sigma, 2: min, 3: max]; frames: [4: mean, 5: sigma, 6: min, 7: max]")
    args = parser.parse_args()
    test_folder = "storage/copies_of_" + args.model + "/tests"
    assert os.path.isdir(test_folder), "File [" + test_folder + "] don't exist!"

    all_frames_data = []
    filenames = find_filenames(test_folder, '.csv')
    sorted_files = [int(_model.split("-")[1]) for _model in filenames]
    sorted_files.sort()
    frames = np.zeros(len(sorted_files))
    for i, frame_number in enumerate(sorted_files):
        filename = test_folder + "/frames-" + str(frame_number) + ".csv"
        assert os.path.isfile(filename), "[" + filename + "] doesn't exist!"
        frames[i] = frame_number
        all_frames = np.loadtxt(filename, delimiter=',', unpack=True)
        all_frames_data.append(np.transpose(all_frames)[2:-2])
    if args.use_3d:
        _plot = create_plot_3d(all_frames_data, frames, args.type_of_result)
    else:
        _plot = create_plot_histograms(all_frames_data, frames, args.type_of_result)
    _plot.show()


def find_filenames(path_to_dir, suffix):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def create_plot_histograms(data, z_frames, acc_type=0):
    infos = np.array(['mean', 'std', 'min', 'max', 'mean', 'std', 'min', 'max'])
    z_label = 'rreturn' if acc_type < 4 else 'frames'
    plt.figure(1)
    i = 1
    _walls = [str(x) for x in np.arange(2, len(data[0]) + 2)]
    nrows = int(len(data) / 4)
    for all_data, frame in zip(data, z_frames):
        plt.subplot(nrows, 4, i)
        _acc = all_data[:, acc_type]
        plt.bar(_walls, _acc)
        if (i - 1) % 4 == 0:
            plt.ylabel(z_label + " " + infos[acc_type])
        if i > (nrows - 1) * 4:
            plt.xlabel("walls")
        plt.title("{0:.0f}".format(frame))
        plt.grid(True)
        plt.gca().set_ylim([0, 1.05 if acc_type < 4 else 2560])
        i += 1
    plt.subplots_adjust(left=0.05, right=0.99, wspace=0.30, hspace=0.4, top=0.95, bottom=0.07)
    plt.rcParams.update({'font.size': 10})
    return plt


def create_plot_3d(data, z_frames, acc_type=0):
    infos = np.array(['mean', 'std', 'min', 'max', 'mean', 'std', 'min', 'max'])
    z_label = 'rreturn' if acc_type < 4 else 'frames'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
    i = 0
    for all_data, z in zip(data, z_frames):
        xs = np.arange(0, all_data.shape[0], 1, dtype=int)
        ys = all_data[:, acc_type]
        ax.bar(xs, ys, zs=z, zdir='y', color=colors[i], alpha=0.8)
        i += 1
    ax.set_xlabel('walls')
    ax.set_ylabel('trains')
    ax.set_zlabel(z_label + " " + infos[acc_type])
    return plt


if __name__ == "__main__":
    main()
