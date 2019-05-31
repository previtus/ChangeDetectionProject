import matplotlib, os
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

import DataLoader, DataPreprocesser, Dataset, Debugger, Settings, ModelHandler, Evaluator
from timeit import default_timer as timer
from datetime import *
import glob
import numpy as np
from random import sample


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

INPUT_FILE_EXCLUSIONS = ""


path = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/"
#star = '*resnet101-8batch_Augmentation1to1_ClassWeights1to3_[KFold_*'
#model_used = "resnet101"
#INPUT_FILE_EXCLUSIONS = "/home/ruzickav/python_projects/ChangeDetectionProject/__OUTPUTS/ResNet101_ManualExclusions.txt"

star = '*resnet50-8batch_Augmentation1to1_ClassWeights1to3_[KFold_*'
model_used = "resnet50"
INPUT_FILE_EXCLUSIONS = "/home/ruzickav/python_projects/ChangeDetectionProject/__OUTPUTS/ResNet50_ManualExclusions.txt"


parser.add_argument('-model_backend', help='Model used in the encoder part of the U-Net structures model', default=model_used)
parser.add_argument('-models_path_star', help='Path to models with reg exp selection', default=path+star)


parser.add_argument('-input_file', help='Alternatively we can have a file with specified input models and their specific settings.', default=INPUT_FILE_EXCLUSIONS)

def main(args):
    print(args)

    threshold_fineness = 0.05 # move this out as a param eventually
    exclusions_by_idxs = []
    selected_model_files = []
    settings = Settings.Settings(args)

    if args.input_file is not "":
        print("")

        with open(args.input_file) as fp:
            line_file = fp.readline()
            line_exclusions = fp.readline()
            cnt = 1
            while line_file:
                print("|"+line_exclusions.strip()+"|"+line_file.strip()+"|")

                selected_model_files.append(line_file.strip())
                if line_exclusions.strip() is not "":
                    exclusions_by_idxs.append(list(map(int,line_exclusions.strip().split(" "))))
                else:
                    exclusions_by_idxs.append([])

                line_file = fp.readline()
                line_exclusions = fp.readline()
                cnt += 1


    else:
        selected_model_files = glob.glob(args.models_path_star)
        selected_model_files.sort()
        exclusions_by_idxs.append([])

    print(exclusions_by_idxs)

    corresponding_fold_indices = []
    corresponding_K_of_folds = []

    print("Selected", len(selected_model_files), "models:")
    for p in selected_model_files:
        print(p)
        print("")
        f = p.split("/")[-1]

        # we will need to get the fold index from the name! (keep it intact)
        assert "[KFold_" in f

        indicator = f.split("[KFold_")[-1]
        limits = indicator.split("z")

        fold_idx = int(limits[0])
        K_of_folds = int(limits[1].split("]")[0])

        #print(fold_idx,"from", K_of_folds,"=", f)
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
        if not os.path.exists("evaluation_plots/"):
            os.makedirs("evaluation_plots/")
        if not os.path.exists(model.model.save_plot_path):
            os.makedirs(model.model.save_plot_path)

        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################

        SimulateUnbalancedDataset = True
        #SimulateUnbalancedDataset = False
        if SimulateUnbalancedDataset:

            [lefts_paths_in_trainAndTest_already, rights_paths, labels_paths] = dataset.paths
            # dataset.train_paths < if we need to be more specific
            print("len(lefts_paths_in_trainAndTest)", len(lefts_paths_in_trainAndTest_already))

            from ActiveLearning.LargeDatasetHandler_AL import get_unbalanced_dataset
            WholeDataset = get_unbalanced_dataset()
            all_left_paths = WholeDataset.paths[0]

            print("We had ", len(lefts_paths_in_trainAndTest_already), " in train+test.")
            print("From ", len(all_left_paths), "all possible pairs in our dataset...")
            allowed_indices = []

            for key_idx in all_left_paths:
                path = all_left_paths[key_idx]

                if path not in lefts_paths_in_trainAndTest_already:
                    allowed_indices.append(key_idx)

            print("... we have", len(allowed_indices), "allowed indices to play with! (which were not in the original train+test)")
            # 81K possibilities!!!

            unbalanced_ratio = 10.0
            unbalanced_ratio = 80.0

            in_test_set_already_N = len(dataset.test[0])
            likely_N_of_changes = in_test_set_already_N / 2.0
            wanted_N_of_nonchanges = int(likely_N_of_changes * unbalanced_ratio)
            print("Sample",wanted_N_of_nonchanges," new non changes...")

            del dataset.train

            import h5py
            """
            def save_images_to_h5(arr, hdf5_path):
                hdf5_file = h5py.File(hdf5_path, mode='w')
                hdf5_file.create_dataset("arr", data=arr, dtype="float32")
                hdf5_file.close()
                print("Saved", len(arr), "images successfully to:", hdf5_path)
                return hdf5_path
            def load_images_from_h5(hdf5_path):
                hdf5_file = h5py.File(hdf5_path, "r")
                arr = hdf5_file['arr'][:]
                hdf5_file.close()
                return arr
            """

            def save_images_to_h5(lefts, rights, labels, hdf5_path):
                SIZE = lefts[0].shape
                SUBSET = len(lefts)
                hdf5_file = h5py.File(hdf5_path, mode='w')
                hdf5_file.create_dataset("lefts", data=lefts, dtype="float32")
                hdf5_file.create_dataset("rights", data=rights, dtype="float32")
                hdf5_file.create_dataset("labels", data=labels, dtype="float32")
                hdf5_file.close()

                print("Saved", SUBSET, "images successfully to:", hdf5_path)

                return hdf5_path

            def load_images_from_h5(hdf5_path):
                hdf5_file = h5py.File(hdf5_path, "r")
                lefts = hdf5_file['lefts'][:]
                rights = hdf5_file['rights'][:]
                labels = hdf5_file['labels'][:]
                hdf5_file.close()

                return lefts, rights, labels

            #path_additional_set = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/datasets/INBALANCED_ADDITIONAL_LEFTS_DATASET_FOR_TESTS8560" # rename to 8560
            #path_additional_set = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/datasets/INBALANCED_ADDITIONAL_LEFTS_DATASET_FOR_TESTS50"
            path_additional_set = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/datasets/INBALANCED_ADDITIONAL_LEFTS_DATASET_FOR_TESTS800"

            PER_BATCH = 800
            batches_to_load = 1 # does mem survive ~ 3200 imgs ??

            # SAVE ONCE, THEN REUSE THOSE BATCHES
            """
            assert False # Do you really want to recalc these? (cca 20 min)
            selected_indices = sample(allowed_indices, wanted_N_of_nonchanges)
            batch_i = 0
            for batch in WholeDataset.generator_for_all_images(PER_BATCH, mode='datalabels', custom_indices_to_sample_from = selected_indices):
                selected_indices, [additional_L, additional_R], additional_V = batch

                save_images_to_h5(additional_L, additional_R, additional_V, path_additional_set+"_"+str(batch_i)+".h5")
                batch_i += 1

                del additional_L
                del additional_R
                del additional_V

                if batch_i >= batches_to_load:
                    break # just 1 batch
            """

            additional_predicted = []
            additional_gts = []

            for i in range(batches_to_load):
                print("loading batch ",i)

                additional_L, additional_R, additional_V = load_images_from_h5(path_additional_set+"_"+str(i)+".h5")
                additional_set = additional_L, additional_R, additional_V
                # goes up to 18G~21G/31G
                additional_set_processed = dataset.dataPreprocesser.apply_on_a_set_nondestructively(additional_set, be_destructive=True)
                add_L, add_R, add_V = additional_set_processed

                if add_L.shape[3] > 3:
                    # 3 channels only - rgb
                    add_L = add_L[:,:,:,1:4]
                    add_R = add_R[:,:,:,1:4]

                print("about to predict batch", i, "with", add_L.shape)
                additional_predicted_batch = model.model.model.predict(x=[add_L, add_R], batch_size=4) # Wait, actually do we create a problem here by moving the BatchNorm stuff in the model?
                # because after it it seems like the model is not predicting the same way
                # try reloading the model afterwards again....
                additional_predicted_batch = additional_predicted_batch[:, :, :, 1]
                additional_gts_batch = add_V
                print("... predicted", len(additional_predicted_batch))

                del add_L
                del add_R

                additional_predicted.extend(additional_predicted_batch)
                additional_gts.extend(additional_gts_batch)

                del additional_predicted_batch
                del additional_gts_batch
                del additional_set_processed
                del additional_set


                ####
                # RESET the model... something has changed in it even if we only predict ... (model's stochasticity ....)

                model_path = selected_model_files[model_idx]
                dataset = Dataset.Dataset(settings) # yo probably slow again ...
                evaluator = Evaluator.Evaluator(settings)
                model = ModelHandler.ModelHandler(settings, dataset)
                model.model.load(model_path)
                folder_name = model_path.split("/")[-1][0:-3]
                model.model.save_plot_path = "evaluation_plots/" + folder_name + "/"

                # PS: different behaviour with the extended set by Unabalanced samples is still possible because of the
                #     way we establish the chosen THR value (as the one which maximizes f1 score)
                #     However ... the human legible outputs should be good - these were done manually on the whole Recall plot curve.
                ####


            additional_predicted = np.asarray(additional_predicted)
            additional_gts = np.asarray(additional_gts)

            print("We have additional predictions:", len(additional_predicted), additional_predicted.shape, "and additional gts:", len(additional_gts), additional_gts.shape)
            optional_additional_predAndGts = [additional_predicted, additional_gts]

        ###############################################################################################################
        ###############################################################################################################
        ###############################################################################################################
        if not SimulateUnbalancedDataset:
            optional_additional_predAndGts = []

        SAVE_ALL_PLOTS = model.model.save_plot_path + "plot"
        statistics = evaluator.unified_test_report([model.model.model], dataset.test, validation_set=dataset.val, postprocessor=model.model.dataPreprocesser,
                                                    name=SAVE_ALL_PLOTS, optionally_save_missclassified=True,
                                                   optional_manual_exclusions = exclusions_by_idxs[model_idx],
                                                   optional_additional_predAndGts = optional_additional_predAndGts)

        # model.model.test(evaluator,show=show,save=save)

        #statistics = model.model.test(evaluator,show=show,save=save, threshold_fineness = threshold_fineness)
        statistics_over_models.append(statistics)

        mask_stats, tiles_stats = statistics

        print("statistics = ",statistics)
        print("mask_stats = ",mask_stats)
        print("tiles_stats = ",tiles_stats)

        del model
        del dataset
        del evaluator
        del optional_additional_predAndGts

        import keras
        keras.backend.clear_session()

    """ debug
    statistics_over_models = [((0.1, 0.609693808104319, 0.782560176106013, 0.9765439612843166, 0.7100948985424048),
      (0.1, 0.8504672897196262, 0.978494623655914, 0.9158878504672897, 0.91)), (
     (0.2, 0.603389186392769, 0.861152404951038, 0.9746090897889895, 0.7240157353295362),
     (0.2, 0.7570093457943925, 1.0, 0.8785046728971962, 0.8617021276595744))]
    """
    add_text = args.model_backend

    statistics_over_models = np.asarray(statistics_over_models)
    if not os.path.exists("evaluation_plots/"):
        os.makedirs("evaluation_plots/")
    np.save("evaluation_plots/statistics_over_models_"+add_text+".npy", statistics_over_models)

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

    file = open("evaluation_plots/report_boxplotStats_"+add_text+".txt", "w")
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
    ax1.set_title('KFoldCrossval statistics (per tiles) - '+settings.model_backend)
    ax1.boxplot(data, labels = xs)
    ax1.set_ylim(0.0,1.0)
    #plt.show()
    plt.savefig("evaluation_plots/boxplot_tiles_stats_"+add_text+".png")
    plt.savefig("evaluation_plots/boxplot_tiles_stats_"+add_text+".pdf")

    fig2, ax2 = plt.subplots()
    ax2.set_title('KFoldCrossval statistics (per masks) - '+settings.model_backend)
    data = [mask_recalls, mask_precisions, mask_accuracies, mask_f1s]
    ax2.boxplot(data, labels = xs)
    ax2.set_ylim(0.0,1.0)
    #plt.show()
    plt.savefig("evaluation_plots/boxplot_masks_stats_"+add_text+".png")
    plt.savefig("evaluation_plots/boxplot_masks_stats_"+add_text+".pdf")

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
