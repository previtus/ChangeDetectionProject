import matplotlib, os
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

import DataLoader, DataPreprocesser, Dataset, Debugger, Settings, ModelHandler, Evaluator
from timeit import default_timer as timer
from datetime import *
import glob
import numpy as np

months = ["unk","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import argparse

parser = argparse.ArgumentParser(description='Project: Change detection on aerial images.')
parser.add_argument('-name', help='run name - will output in this dir', default="Run-"+month+"-"+day)
#parser.add_argument('-KFOLDS', help='Number of folds', default='10')
#parser.add_argument('-FOLD_I', help='This fold i', default='0')
parser.add_argument('-train_epochs', help='How many epochs', default='100')
parser.add_argument('-train_batch', help='How big batch size', default='8')


model_used = "resnet101"
parser.add_argument('-model_backend', help='Model used in the encoder part of the U-Net structures model', default=model_used)

path = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/"
star = '*resnet101-8batch_Augmentation1to1_ClassWeights1to3_[KFold_*'

parser.add_argument('-models_path_star', help='Path to models with reg exp selection', default=path+star)

def main(args):
    print(args)

    threshold_fineness = 0.05 # move this out as a param eventually

    settings = Settings.Settings(args)

    selected_model_files = glob.glob(args.models_path_star)
    selected_model_files.sort()
    corresponding_fold_indices = []
    corresponding_K_of_folds = []

    print("Selected", len(selected_model_files), "models:")
    for p in selected_model_files:
        f = p.split("/")[-1]

        # we will need to get the fold index from the name! (keep it intact)
        assert "[KFold_" in f

        indicator = f.split("[KFold_")[-1]
        limits = indicator.split("z")

        fold_idx = int(limits[0])
        K_of_folds = int(limits[1].split("]")[0])

        print(fold_idx,"from", K_of_folds,"=", f)
        corresponding_fold_indices.append(fold_idx)
        corresponding_K_of_folds.append(K_of_folds)


    print("We got these indices of folds", corresponding_fold_indices)
    print("And these K values for kfoldcrossval", corresponding_K_of_folds)

    # TEST MODELS ONE BY ONE
    statistics_over_models = []

    for model_idx in range(len(selected_model_files)):
        #for model_idx in range(2):
        model_path = selected_model_files[model_idx]
        settings.TestDataset_Fold_Index = corresponding_fold_indices[model_idx]
        settings.TestDataset_K_Folds = corresponding_K_of_folds[model_idx]

        assert settings.TestDataset_Fold_Index < settings.TestDataset_K_Folds
        print(model_path)

        dataset = Dataset.Dataset(settings)
        evaluator = Evaluator.Evaluator(settings)

        show = False
        save = True

        #dataset.dataset
        settings.model_backend = args.model_backend
        settings.train_epochs = int(args.train_epochs)
        settings.train_batch = int(args.train_batch)
        model = ModelHandler.ModelHandler(settings, dataset)

        # K-Fold_Crossval:
        ####model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_"+model_txt+"_["+kfold_txt+"].h5")
        model.model.load(model_path)

        folder_name = model_path.split("/")[-1][0:-3]
        model.model.save_plot_path = "evaluation_plots/" + folder_name + "/"
        import os
        if not os.path.exists(model.model.save_plot_path):
            os.makedirs(model.model.save_plot_path)

        statistics = model.model.test(evaluator,show=show,save=save, threshold_fineness = threshold_fineness)
        statistics_over_models.append(statistics)

        mask_stats, tiles_stats = statistics

        print("statistics = ",statistics)
        print("mask_stats = ",mask_stats)
        print("tiles_stats = ",tiles_stats)

        del model
        del dataset
        del evaluator

        import keras
        keras.backend.clear_session()

    """ debug
    statistics_over_models = [((0.1, 0.609693808104319, 0.782560176106013, 0.9765439612843166, 0.7100948985424048),
      (0.1, 0.8504672897196262, 0.978494623655914, 0.9158878504672897, 0.91)), (
     (0.2, 0.603389186392769, 0.861152404951038, 0.9746090897889895, 0.7240157353295362),
     (0.2, 0.7570093457943925, 1.0, 0.8785046728971962, 0.8617021276595744))]
    """

    statistics_over_models = np.asarray(statistics_over_models)
    np.save("evaluation_plots/statistics_over_models.npy", statistics_over_models)

    #####statistics_over_models = np.load("evaluation_plots/resnet101_kfolds_0to8.npy")

    ### Process overall statistics -> boxplots!
    print("Overall statistics::: (",len(statistics_over_models),")")
    print(statistics_over_models)

    # each model has [mask_stats, tiles_stats] = [[thr, recall, precision, accuracy, f1], [...]]
    thresholds = []

    tiles_recalls = []
    tiles_precisions = []
    tiles_accuracies = []
    tiles_f1s = []

    mask_recalls = []
    mask_precisions = []
    mask_accuracies = []
    mask_f1s = []

    for stats in statistics_over_models:
        mask_stats, tiles_stats = stats
        thresholds.append(mask_stats[0])

        tiles_recalls.append(tiles_stats[1])
        mask_recalls.append(mask_stats[1])

        tiles_precisions.append(tiles_stats[2])
        mask_precisions.append(mask_stats[2])

        tiles_accuracies.append(tiles_stats[3])
        mask_accuracies.append(mask_stats[3])

        tiles_f1s.append(tiles_stats[4])
        mask_f1s.append(mask_stats[4])


    # REPORT
    report_text = ""
    report_text += "Tiles evaluation:\n"
    report_text += "mean tiles_recalls = " + str( 100.0 * np.mean(tiles_recalls) ) + " +- " + str( 100.0 *np.std(tiles_recalls) ) + " std \n"
    report_text += "mean tiles_precisions = " + str( 100.0 *np.mean(tiles_precisions) ) + " +- " + str( 100.0 *np.std(tiles_precisions) ) + " std \n"
    report_text += "mean tiles_accuracies = " + str( 100.0 *np.mean(tiles_accuracies) ) + " +- " + str( 100.0 *np.std(tiles_accuracies) ) + " std \n"
    report_text += "mean tiles_f1s = " + str( 100.0 *np.mean(tiles_f1s) ) + " +- " + str( 100.0 *np.std(tiles_f1s) ) + " std \n"
    report_text += "\n"
    report_text += "Mask evaluation:\n"
    report_text += "mean mask_recalls = " + str( 100.0 *np.mean(mask_recalls) ) + " +- " + str( 100.0 *np.std(mask_recalls) ) + " std \n"
    report_text += "mean mask_precisions = " + str( 100.0 *np.mean(mask_precisions) ) + " +- " + str( 100.0 *np.std(mask_precisions) ) + " std \n"
    report_text += "mean mask_accuracies = " + str( 100.0 *np.mean(mask_accuracies) ) + " +- " + str( 100.0 *np.std(mask_accuracies) ) + " std \n"
    report_text += "mean mask_f1s = " + str( 100.0 *np.mean(mask_f1s) ) + " +- " + str( 100.0 *np.std(mask_f1s) ) + " std \n"

    file = open("evaluation_plots/report_boxplotStats.txt", "w")
    file.write(report_text)
    file.close()

    xs = ["recall", "precision", "accuracy", "f1"]
    data = [tiles_recalls, tiles_precisions, tiles_accuracies, tiles_f1s]

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    """
    ax = sns.boxplot(x=xs, y=data)
    ax.set_title('Stats per tiles')
    ax.set_ylim(0.0,1.0)
    plt.show()
    """

    fig1, ax1 = plt.subplots()
    ax1.set_title('KFoldCrossval statistics (per tiles) - ResNet101')
    ax1.boxplot(data, labels = xs)
    ax1.set_ylim(0.0,1.0)
    #plt.show()
    plt.savefig("evaluation_plots/boxplot_tiles_stats.png")
    plt.savefig("evaluation_plots/boxplot_tiles_stats.pdf")

    fig2, ax2 = plt.subplots()
    ax2.set_title('KFoldCrossval statistics (per masks) - ResNet101')
    data = [mask_recalls, mask_precisions, mask_accuracies, mask_f1s]
    ax2.boxplot(data, labels = xs)
    ax2.set_ylim(0.0,1.0)
    #plt.show()
    plt.savefig("evaluation_plots/boxplot_masks_stats.png")
    plt.savefig("evaluation_plots/boxplot_masks_stats.pdf")

    print("Just as an additional info, these were the chosen thresholds across models:", thresholds)

if __name__ == '__main__':
    args = parser.parse_args()

    start = timer()

    print("### EVALUATION OF LOADED TRAINED MODEL ###")
    main(args)

    end = timer()
    time = (end - start)

    print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")
    # for 9 models of resnet101 this took 26 minutes!

    import keras
    keras.backend.clear_session()
