# === Initialize sets - Unlabeled, Train and Test
import matplotlib, os
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

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

months = ["unk", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import argparse

parser = argparse.ArgumentParser(description='Project: Deep Active Learning for Change detection on aerial images.')
parser.add_argument('-name', help='run name - will output in this dir', default="Run-"+month+"-"+day)

parser.add_argument('-model_epochs', help='How many epochs will each model train?', default="50") #50? 100?
parser.add_argument('-model_batchsize', help='How big batch size for each model? (This is limited by the GPUs memory)', default="4")
parser.add_argument('-model_backbone', help='Encoder', default="resnet34")

parser.add_argument('-train_augmentation', help='Turn on augmentation? (one new image for each image in the train set - effectively doubling the dataset size)', default="False")

parser.add_argument('-AL_iterations', help='Number of iterations in the Active Learning loop', default="10")
parser.add_argument('-AL_initialsample_size', help='Start with this many sampled images in the training set', default="100")
parser.add_argument('-AL_testsample_size', help='Have this many balanced sample images in the testing set (used for plots)', default="200")
parser.add_argument('-AL_valsample_size', help='Have this many balanced sample images in the validation set (used for automatic thr choice and val errs)', default="200")
parser.add_argument('-AL_iterationsample_size', help='Add this many images in each iteration', default="100")

parser.add_argument('-AL_method', help='Sampling method (choose from "Random", "Ensemble", "MonteCarloBatchNormalization")', default="Random")

parser.add_argument('-AL_AcquisitionFunction', help='For any method other than Random (choose from "Variance", "Entropy", "BALD")', default="Variance")

parser.add_argument('-AL_Ensemble_numofmodels', help='If we chose Ensemble, how many models are there?', default="3")

parser.add_argument('-DEBUG_remove_from_dataset', help='Debug to remove random samples without change from the original dataset...', default="40000")
parser.add_argument('-DEBUG_loadLastALModels', help='Debug function - load last saved model weights instead of training ...', default="False")



if __name__ == '__main__':
    args = parser.parse_args()

    start = timer()

    from ActiveLearning.ActiveLearningLoop import main as al_main
    al_main(args)

    end = timer()
    time = (end - start)

    print("This run took " + str(time) + "s (" + str(time / 60.0) + "min)")

    import keras

    keras.backend.clear_session()
