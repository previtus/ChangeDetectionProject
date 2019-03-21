import DataLoader, DataPreprocesser, Dataset, Debugger, Settings, ModelHandler, Evaluator
from timeit import default_timer as timer
from datetime import *

months = ["unk","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import argparse

parser = argparse.ArgumentParser(description='Project: Change detection on aerial images.')

def main(args):
    print(args)

    settings = Settings.Settings(args)
    dataset = Dataset.Dataset(settings)
    evaluator = Evaluator.Evaluator(settings)

    #dataset.dataset
    model = ModelHandler.ModelHandler(settings, dataset)

    #model.model.train()

    # clean 2 - manual cleaning, only 256x256 with 32px overlap ("256_cleanManual")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_cleanManual_256dataset_WeightsT1.h5") # is 1 to 100


    model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_cleanManual_256dataset_WeightsT1_1to3w.h5")#1 to 3, 30 ep
    #as a sanity check
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_cleanManual_256dataset_WeightsT1_1to1w.h5")#1 to 1, 30 ep



    # clean 1 - keeping in only polygons with area 40 and bigger
    # Re-Check paths here!

    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_clean256dataset_.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_clean256dataset_.h5")

    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_clean112dataset_.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_clean112dataset_.h5")


    # first "unclean" dataset
    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_full256dataset_.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_full256dataset_.h5")

    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_full112dataset_.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1b_full112dataset_.h5")

    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1_full112dataset_300ep_overfit.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1_full112dataset_300ep_overfit.h5")

    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1_full112dataset_DataAug.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1_full112dataset_DataAug.h5") # 100 epochs, on normalized L/R images + data augmentation

    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1_full112dataset_REDOS.h5")

    # softmax
    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1_full112dataset_.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1_full112dataset_.h5") # 100 epochs, on normalized L/R images

    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1_full256dataset_.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel1_full256dataset_.h5") # 100 epochs, on normalized L/R images

    # sigmoid
    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/onera_weights_Take2.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/onera_weights_Take2.h5")
    #model.model.load("/home/ruzickav/python_projects/test1/last_OSCD_model_weightsNewer.h5")

    model.model.test(evaluator)
    #model.model.test_show_on_train_data_to_see_overfit(evaluator)

if __name__ == '__main__':
    args = parser.parse_args()

    start = timer()

    args.name = "Run"+day+"-"+month

    main(args)

    end = timer()
    time = (end - start)
    #print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")

