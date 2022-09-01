
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from src.utils import check_exist_and_create, load_json, save_dict_to_json

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/test_case_nonsep',
                    help="Directory containing 'test' ")
parser.add_argument('--mode', default='debug',
                    help="mode debug keeps more detail; mode paper is clean' ")
parser.add_argument('--sudoku', default=True, action='store_true',
                    help="while to plot sudoku")
parser.add_argument('--force_overwrite', default=False, action='store_true',
                    help="For debug. Force to clean the 'figure' folder each running ")

def plot_heat_map(variable, title, savepath):
    plt.figure(figsize=(8,6))
    sns.heatmap(variable)
    plt.title(title)
    plt.savefig(os.path.join(savepath, title+".png"), dpi=300)

def main(experiment_dir):
    viz_path = os.path.join(experiment_dir, "viz")
    check_exist_and_create(viz_path)
    test_result_folders = os.listdir(os.path.join(experiment_dir, "test_result"))

    for test_result_folder in test_result_folders:
        savepath = os.path.join(viz_path, test_result_folder)
        check_exist_and_create(savepath)
        for file in os.listdir(os.path.join(experiment_dir, "test_result", test_result_folder)):
            name, ext = os.path.splitext(file)
            if ext == ".csv":
                variable = np.loadtxt(os.path.join(experiment_dir, "test_result", test_result_folder, file),
                                      delimiter=",")
                plot_heat_map(variable, name, savepath)


if __name__ == "__main__":
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    main(experiment_dir)



