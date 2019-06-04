# === Initialize sets - Unlabeled, Train and Test
import keras
from ActiveLearning.LargeDatasetHandler_AL import get_balanced_dataset, get_unbalanced_dataset
from ActiveLearning.LargeDatasetHandler_AL import LargeDatasetHandler_AL
from ActiveLearning.ModelHandler_dataIndependent import ModelHandler_dataIndependent
from ActiveLearning.DataPreprocesser_dataIndependent import DataPreprocesser_dataIndependent
from ActiveLearning.TrainTestHandler import TrainTestHandler
from Evaluator import Evaluator
from timeit import default_timer as timer
from datetime import *
import os

months = ["unk","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import argparse

parser = argparse.ArgumentParser(description='Project: Deep Active Learning for Change detection on aerial images.')
parser.add_argument('-name', help='run name - will output in this dir', default="Run-"+month+"-"+day)

parser.add_argument('-model_epochs', help='How many epochs will each model train?', default="50")
parser.add_argument('-model_batchsize', help='How big batch size for each model? (This is limited by the GPUs memory)', default="4")

parser.add_argument('-train_augmentation', help='Turn on augmentation? (one new image for each image in the train set - effectively doubling the dataset size)', default="False")

parser.add_argument('-AL_iterations', help='Number of iterations in the Active Learning loop', default="10")
parser.add_argument('-AL_initialsample_size', help='Start with this many sampled images in the training set', default="100")
parser.add_argument('-AL_testsample_size', help='Have this many balanced sample images in the testing set', default="250")
parser.add_argument('-AL_iterationsample_size', help='Add this many images in each iteration', default="100")

parser.add_argument('-AL_method', help='Sampling method (choose from "Random", "Ensemble")', default="Random")

parser.add_argument('-AL_Ensemble_numofmodels', help='If we chose Ensemble, how many models are there?', default="3")

parser.add_argument('-DEBUG_remove_from_dataset', help='Debug to remove random samples without change from the original dataset...', default="40000")

def main(args):
    args_txt = str(args)
    print(args_txt)

    INITIAL_SAMPLE_SIZE = int(args.AL_initialsample_size) #100
    TEST_SAMPLE_SIZE = int(args.AL_testsample_size) #250
    import matplotlib.pyplot as plt
    import numpy as np

    args.name = "May31EnsembleFullUnbalanced20It_AugOn"
    path_to_the_corresponding_statistics_file = "/home/ruzickav/python_projects/ChangeDetectionProject/ActiveLearning/_June1st_longRunResults/__serverRuns/_(unfinished)/[May31EnsembleFullUnbalanced20It_AugOn]_al_statistics.npy"

    statistics = np.load(path_to_the_corresponding_statistics_file)

    pixel_statistics, tile_statistics = statistics
    pixels_thresholds, xs_number_of_data, pixels_recalls, pixels_precisions, pixels_accuracies, pixels_f1s, Ns_changed, Ns_nochanged, pixels_AUCs = pixel_statistics
    tiles_thresholds, xs_number_of_data, tiles_recalls, tiles_precisions, tiles_accuracies, tiles_f1s, Ns_changed, Ns_nochanged = tile_statistics

    # PIXELS
    print("pixels_thresholds:", pixels_thresholds)

    print("xs_number_of_data:", xs_number_of_data)
    print("pixels_recalls:", pixels_recalls)
    print("pixels_precisions:", pixels_precisions)
    print("pixels_accuracies:", pixels_accuracies)
    print("pixels_f1s:", pixels_f1s)
    print("pixels_AUCs:", pixels_AUCs)

    print("Ns_changed:", Ns_changed)
    print("Ns_nochanged:", Ns_nochanged)

    plt.figure(figsize=(7, 7)) # w, h
    plt.plot(xs_number_of_data, pixels_thresholds, color='black', label="thresholds")

    plt.plot(xs_number_of_data, pixels_recalls, color='red', marker='o', label="recalls")
    plt.plot(xs_number_of_data, pixels_precisions, color='blue', marker='o', label="precisions")
    plt.plot(xs_number_of_data, pixels_accuracies, color='green', marker='o', label="accuracies")
    plt.plot(xs_number_of_data, pixels_f1s, color='orange', marker='o', label="f1s")
    plt.plot(xs_number_of_data, pixels_AUCs, color='magenta', marker='o', label="AUCs")

    #plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)

    plt.legend()
    plt.ylim(0.0, 1.0)

    plt.savefig("["+args.name+"]_dbg_last_al_big_plot_pixelsScores.png")
    plt.close()

    # TILES
    print("tiles_thresholds:", tiles_thresholds)

    print("xs_number_of_data:", xs_number_of_data)
    print("tiles_recalls:", tiles_recalls)
    print("tiles_precisions:", tiles_precisions)
    print("tiles_accuracies:", tiles_accuracies)
    print("tiles_f1s:", tiles_f1s)

    print("Ns_changed:", Ns_changed)
    print("Ns_nochanged:", Ns_nochanged)

    plt.figure(figsize=(7, 7)) # w, h
    plt.plot(xs_number_of_data, tiles_thresholds, color='black', label="thresholds")

    plt.plot(xs_number_of_data, tiles_recalls, color='red', marker='o', label="recalls")
    plt.plot(xs_number_of_data, tiles_precisions, color='blue', marker='o', label="precisions")
    plt.plot(xs_number_of_data, tiles_accuracies, color='green', marker='o', label="accuracies")
    plt.plot(xs_number_of_data, tiles_f1s, color='orange', marker='o', label="f1s")

    #plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)

    plt.legend()
    plt.ylim(0.0, 1.0)

    plt.savefig("["+args.name+"]_dbg_last_al_big_plot_tilesScores.png")
    plt.close()


    plt.figure(figsize=(7, 7)) # w, h

    N = len(Ns_changed)
    ind = np.arange(N)
    width = 0.35
    p1 = plt.bar(ind, Ns_changed, width)
    p2 = plt.bar(ind, Ns_nochanged, width, bottom=Ns_changed)

    plt.ylabel('number of data samples')
    plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)
    plt.legend((p1[0], p2[0]), ('Change', 'NoChange'))

    plt.savefig("["+args.name+"]_dbg_last_al_balance_plot.png")
    plt.close()




if __name__ == '__main__':
    args = parser.parse_args()

    start = timer()

    main(args)

    end = timer()
    time = (end - start)

    print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")

    import keras
    keras.backend.clear_session()
