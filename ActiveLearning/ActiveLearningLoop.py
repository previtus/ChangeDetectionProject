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
parser.add_argument('-model_backbone', help='Encoder', default="resnet34")

parser.add_argument('-train_augmentation', help='Turn on augmentation? (one new image for each image in the train set - effectively doubling the dataset size)', default="False")

parser.add_argument('-AL_iterations', help='Number of iterations in the Active Learning loop', default="10")
parser.add_argument('-AL_initialsample_size', help='Start with this many sampled images in the training set', default="100")
parser.add_argument('-AL_testsample_size', help='Have this many balanced sample images in the testing set (used for plots)', default="200")
parser.add_argument('-AL_valsample_size', help='Have this many balanced sample images in the validation set (used for automatic thr choice and val errs)', default="200")
parser.add_argument('-AL_iterationsample_size', help='Add this many images in each iteration', default="100")

parser.add_argument('-AL_method', help='Sampling method (choose from "Random", "Ensemble")', default="Random")

parser.add_argument('-AL_Ensemble_numofmodels', help='If we chose Ensemble, how many models are there?', default="3")

parser.add_argument('-DEBUG_remove_from_dataset', help='Debug to remove random samples without change from the original dataset...', default="40000")
parser.add_argument('-DEBUG_loadLastALModels', help='Debug function - load last saved model weights instead of training ...', default="False")

def main(args):
    args_txt = str(args)
    print(args_txt)

    import random
    import numpy
    seed_num = 30
    random.seed(seed_num) # samples
    numpy.random.seed(seed_num) # shuffles
    # keras and it's training is not influenced by this

    #import tensorflow as tf
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True
    #sess = tf.Session(config=config)

    """ TRAINING """
    epochs = int(args.model_epochs) # 50
    batch_size = int(args.model_batchsize) # 4
    augmentation = (args.train_augmentation == 'True') #False
    model_backbone = "resnet34"


    """ ACTIVE LEARNING """
    N_ITERATIONS = int(args.AL_iterations) #10
    # we have approx this ratio in our data 1072 : 81592

    INITIAL_SAMPLE_SIZE = int(args.AL_initialsample_size) #100
    TEST_SAMPLE_SIZE = int(args.AL_testsample_size) #200
    VAL_SAMPLE_SIZE = int(args.AL_valsample_size) #200
    ITERATION_SAMPLE_SIZE = int(args.AL_iterationsample_size) #100


    acquisition_function_mode = args.AL_method #"Ensemble" / "Random"
    ModelEnsemble_N = int(args.AL_Ensemble_numofmodels)

    if acquisition_function_mode is not "Ensemble":
        ModelEnsemble_N = 1

    ENSEMBLE_tmp_how_many_from_random = 0 # Hybridization


    REMOVE_SIZE = int(args.DEBUG_remove_from_dataset)
    DEBUG_loadLastALModels = (args.DEBUG_loadLastALModels == 'True')
    DEBUG_skip_evaluation = False
    threshold_fineness = 0.05 # default

    # New feature, failsafe for models in Ensembles ...
    FailSafeON = True # default
    FailSafe__ValLossThr = 1.0 # default # maybe should be 1.5 ???

    # LOCAL OVERRIDES:
    """
    acquisition_function_mode = "Ensemble"
    ModelEnsemble_N = 1
    INITIAL_SAMPLE_SIZE = 600
    N_ITERATIONS = 2
    epochs = 35
    REMOVE_SIZE = 75000
    DEBUG_loadLastALModels = True
    DEBUG_skip_evaluation = True # won't save the plots, but jumps directly to the AL acquisition functions
    
    threshold_fineness = 0.1 # 0.05 makes nicer plots
    """
    ## Loop starts with a small train set (INITIAL_SAMPLE_SIZE)
    ## then adds some number every iteration (ITERATION_SAMPLE_SIZE)
    ## also every iteration tests on a selected test sample

    import matplotlib.pyplot as plt
    import numpy as np


    #in_memory = False
    in_memory = True # when it all fits its much faster
    #RemainingUnlabeledSet = get_balanced_dataset(in_memory)
    RemainingUnlabeledSet = get_unbalanced_dataset()

    # HAX
    print("-----------------------------")
    print("HAX: REMOVING SAMPLES SO WE DON'T GO MENTAL! (80k:1k ratio now 40k:1k ... still a big difference ...")
    #REMOVE_SIZE = 70000 # super HAX
    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset_balanced_classes(REMOVE_SIZE, 0.0)
    removed_items = RemainingUnlabeledSet.pop_items(selected_indices)

    RemainingUnlabeledSet.report()
    print("-----------------------------")

    # TODO: ALLOW LOADING TEST+VAL FROM PRECOMPUTED h5 FILES TOO (for server deplyment)
    # TODO: INITIAL TRAIN SET also needs it
    # TODO: so .... allow loading from the Chunks

    settings = RemainingUnlabeledSet.settings
    TestSet = LargeDatasetHandler_AL(settings, "inmemory")
    #selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(TEST_SAMPLE_SIZE)
    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset_balanced_classes(TEST_SAMPLE_SIZE, 0.5) # should be balanced
    print("test set indices >> ", selected_indices)
    packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
    TestSet.add_items(packed_items)
    print("Test set:")
    TestSet.report()
    N_change_test, N_nochange_test = TestSet.report_balance_of_class_labels()
    print("are we balanced in the test set?? Change:NoChange",N_change_test, N_nochange_test)



    ValSet = LargeDatasetHandler_AL(settings, "inmemory")
    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset_balanced_classes(VAL_SAMPLE_SIZE, 0.5) # should be balanced
    packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
    ValSet.add_items(packed_items)
    print("Validation set:")
    ValSet.report()
    N_change_val, N_nochange_val = TestSet.report_balance_of_class_labels()
    print("are we balanced in the val set?? Change:NoChange",N_change_val, N_nochange_val)



    TrainSet = LargeDatasetHandler_AL(settings, "inmemory")

    # Initial bump of data
    #selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(INITIAL_SAMPLE_SIZE)
    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset_balanced_classes(INITIAL_SAMPLE_SIZE, 0.5) # possibly also should be balanced?

    packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
    TrainSet.add_items(packed_items)

    # === Model and preprocessors

    test_data, test_paths, _ = TestSet.get_all_data_as_arrays()
    val_data, val_paths, _ = ValSet.get_all_data_as_arrays()

    dataPreprocesser = DataPreprocesser_dataIndependent(settings, number_of_channels=4)
    trainTestHandler = TrainTestHandler(settings)
    evaluator = Evaluator(settings)

    # === Start looping:
    tiles_recalls = []
    tiles_precisions = []
    tiles_accuracies = []
    tiles_f1s = []

    tiles_thresholds = []

    pixels_recalls = []
    pixels_precisions = []
    pixels_accuracies = []
    pixels_f1s = []
    pixels_AUCs = []

    pixels_thresholds = []

    xs_number_of_data = []
    Ns_changed = []
    Ns_nochanged = []

    #for active_learning_iteration in range(30): # 30*500 => 15k images
    for active_learning_iteration in range(N_ITERATIONS):
        xs_number_of_data.append(TrainSet.get_number_of_samples())
        # survived to 0-5 = 6*500 = 3000 images (I think that then the RAM was filled ...)

        # Save args log:
        DEBUG_SAVE_ALL_THR_FOLDER = "["+args.name+"]_iteration_"+str(active_learning_iteration).zfill(2)+"_debugThrOverview/"
        DEBUG_SAVE_ALL_THR_PLOTS = DEBUG_SAVE_ALL_THR_FOLDER+"iteration_"+str(active_learning_iteration).zfill(2)
        #DEBUG_SAVE_ALL_THR_PLOTS = None
        if not os.path.exists(DEBUG_SAVE_ALL_THR_FOLDER):
            os.makedirs(DEBUG_SAVE_ALL_THR_FOLDER)
        file = open(DEBUG_SAVE_ALL_THR_FOLDER+"args.txt", "w")
        file.write(args_txt)
        file.close()

        print("\n")
        print("==========================================================")
        print("\n")

        print("Active Learning iteration:", active_learning_iteration)
        start = timer()

        print("Unlabeled:")
        RemainingUnlabeledSet.report()
        print("Train:")
        TrainSet.report()

        # what about the balance of the data?

        N_change, N_nochange = TrainSet.report_balance_of_class_labels()
        Ns_changed.append(N_change)
        Ns_nochanged.append(N_nochange)

        train_data, train_paths, _ = TrainSet.get_all_data_as_arrays()
        print("fitting the preprocessor on this dataset")

        dataPreprocesser.fit_on_train(train_data)

        # non destructive copies of the sets
        processed_train = dataPreprocesser.apply_on_a_set_nondestructively(train_data)
        processed_test = dataPreprocesser.apply_on_a_set_nondestructively(test_data)
        processed_val = dataPreprocesser.apply_on_a_set_nondestructively(val_data)

        print("Train shapes:", processed_train[0].shape, processed_train[1].shape, processed_train[2].shape)
        print("Test shapes:", processed_test[0].shape, processed_test[1].shape, processed_test[2].shape)
        print("Val shapes:", processed_val[0].shape, processed_val[1].shape, processed_val[2].shape)

        # Init model
        #ModelHandler = ModelHandler_dataIndependent(settings, BACKBONE=model_backbone)
        #model = ModelHandler.model

        ModelEnsemble = []
        Separate_Train_Eval__TrainAndSave = True # < we want to train new ones every iteration
        TMP__DontSave = False
        TMP__DontSave = True # < dont save
        Separate_Train_Eval__TrainAndSave = False # < this loads from the last interation (kept in case of mem errs)

        Separate_Train_Eval__TrainAndSave = not DEBUG_loadLastALModels # True in the command line means loading the file ...

        for i in range(ModelEnsemble_N):
            modelHandler = ModelHandler_dataIndependent(settings, BACKBONE=model_backbone)
            ModelEnsemble.append(modelHandler)
            # These models have the same encoder part (same weights as loaded from the imagenet pretrained model)
            # ... however their decoder parts are initialized differently.

        """
        # Report on initial weights of a model:
        model1 = ModelEnsemble[0]
        model2 = ModelEnsemble[1]
    
        for l1, l2 in zip(model1.layers[2].layers, model2.layers[2].layers):
            w1 = l1.get_weights()
            w2 = l2.get_weights()
            print(l1.name, np.array_equal(w1,w2))
        """

        #print("All layers (i hope that this gets inside the embedded model toooo...)")
        #for layer in model.layers:
        #    print(layer.name, layer.output_shape)
        #model.layers[2].summary()

        if Separate_Train_Eval__TrainAndSave:
            # === Train!:
            print("Now I would train ...")
            for i in range(ModelEnsemble_N):
                print("Training model",i,"in the ensemble (out of",ModelEnsemble_N,"):")

                AllowedFailsafeRetrains = 4 # Not to actually get stuck
                failed_training_flag = True # When looking at results on VAL set (don't know what would happen on TEST).
                while failed_training_flag:

                    history, broken_flag, failed_training_flag = trainTestHandler.train(ModelEnsemble[i].model, processed_train, processed_val, epochs, batch_size,
                                    augmentation = augmentation, DEBUG_POSTPROCESSER=dataPreprocesser, name=DEBUG_SAVE_ALL_THR_PLOTS+"_model_"+str(i),
                                    FailSafeON=FailSafeON, FailSafe__ValLossThr=FailSafe__ValLossThr)
                    # Maybe make a fail safe here in case the final VAL_LOSS  is too large -> reinit and retrain!
                    # Or if the recall is basically 0.0 (failed case when the model trains to mark everything as no-change

                    if failed_training_flag and AllowedFailsafeRetrains > 0:
                        # ??? and active_learning_iteration > 0:
                        # ??? skip 0th iteration # ... discussable as well
                        # this failsafe works with all methods

                        AllowedFailsafeRetrains -= 1

                        print("Detected model training fail safe - revert the model - then retrain!")
                        modelHandler_restart = ModelHandler_dataIndependent(settings, BACKBONE=model_backbone)
                        ModelEnsemble[i] = modelHandler_restart

                if broken_flag:
                    print("Got as far as until AL iteration:", active_learning_iteration, " ... now breaking")
                    break

            if not TMP__DontSave:
                root = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/models_in_al/"
                # now save these models
                for i in range(ModelEnsemble_N):
                    modelHandler = ModelEnsemble[i]
                    modelHandler.save(root+"initTests_"+str(i))

        else:
            print("!!! Loading last saved models (with fixed paths, so this might break!)\n!!! BE SURE YOU WANT THIS.")
            # Load and then eval! (check stuff first ...)
            root = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/models_in_al/"
            for i in range(ModelEnsemble_N):
                modelHandler = ModelEnsemble[i]
                modelHandler.load(root+"initTests_"+str(i))


        if not DEBUG_skip_evaluation:
            print("Now I would test ...")

            models = []
            for i in range(ModelEnsemble_N):
                models.append( ModelEnsemble[i].model )

            statistics = evaluator.unified_test_report(models, processed_test, validation_set=processed_val, postprocessor=dataPreprocesser,
                                                       name=DEBUG_SAVE_ALL_THR_PLOTS,
                                                       optionally_save_missclassified=True, threshold_fineness = threshold_fineness)

            mask_stats, tiles_stats = statistics
            tiles_best_thr, tiles_selected_recall, tiles_selected_precision, tiles_selected_accuracy, tiles_selected_f1 = tiles_stats
            pixels_best_thr, pixels_selected_recall, pixels_selected_precision, pixels_selected_accuracy, pixels_selected_f1, pixels_auc = mask_stats

            pixels_recalls.append(pixels_selected_recall)
            pixels_precisions.append(pixels_selected_precision)
            pixels_accuracies.append(pixels_selected_accuracy)
            pixels_f1s.append(pixels_selected_f1)
            pixels_AUCs.append(pixels_auc)
            pixels_thresholds.append(pixels_best_thr)

            tiles_recalls.append(tiles_selected_recall)
            tiles_precisions.append(tiles_selected_precision)
            tiles_accuracies.append(tiles_selected_accuracy)
            tiles_f1s.append(tiles_selected_f1)
            tiles_thresholds.append(tiles_best_thr)

        print("Add new samples as per AL paradigm ... acquisition_function_mode=",acquisition_function_mode)

        # this is not necessary for Random ...
        if acquisition_function_mode == "Ensemble":

            entropies_over_samples = []
            variances_over_samples = []
            overall_indices = np.asarray([]) # important to concat these correctly across batches

            # BATCH THIS ENTIRE SEGMENT
            PER_BATCH = 2048 # Depends on memory really ...

            # Yields a large batch sample
            #for batch in RemainingUnlabeledSet.generator_for_all_images(PER_BATCH, mode='dataonly'): <<<<< This is faster if we are on local PC and load relatively small amount of images.
            for batch in RemainingUnlabeledSet.generator_for_all_images(PER_BATCH, mode='dataonly_LOADBATCHFILES'): # Yields a large batch sample
                remaining_indices = batch[0]
                if len(remaining_indices) == 0:
                    print("Everything from this batch was deleted (during loading batchfiles), skipping to the next one...")
                    continue
                remaining_data = batch[1]  # lefts and rights, no labels!
                print("report on sizes of the batch = len(remaining_indices)", len(remaining_indices), "len(left)=len(remaining_data[0])=", len(remaining_data[0]))

                print("MegaBatch (",len(entropies_over_samples),"/",RemainingUnlabeledSet.N_of_data,") of size ", len(remaining_indices), " = indices from id", remaining_indices[0], "to id", remaining_indices[-1])
                processed_remaining = dataPreprocesser.apply_on_a_set_nondestructively(remaining_data, no_labels=True)

                # have the models in ensemble make predictions
                # Function EnsemblePrediction() maybe even in batches?
                PredictionsEnsemble = []

                ##### potentially problematic -> too big for mem!
                #### remaining_data, remaining_paths, remaining_indices = RemainingUnlabeledSet.get_all_data_as_arrays()
                #### processed_remaining = dataPreprocesser.apply_on_a_set_nondestructively(remaining_data)

                for nth, handler in enumerate(ModelEnsemble):
                    model = handler.model

                    test_L, test_R = processed_remaining

                    if test_L.shape[3] > 3:
                        # 3 channels only - rgb
                        test_L = test_L[:,:,:,1:4]
                        test_R = test_R[:,:,:,1:4]
                    # label also reshape

                    # only 5 images now!
                    #subs = 100
                    #test_L = test_L[0:subs]
                    #test_R = test_R[0:subs]

                    print("Predicting for", nth, "model in the ensemble - for disagreement calculations (on predicted size=",len(test_L),")")
                    predicted = model.predict(x=[test_L, test_R], batch_size=4)
                    predicted = predicted[:, :, :, 1] # 2 channels of softmax with 2 classes is useless - use just one

                    PredictionsEnsemble.append(predicted)

                from scipy.stats import entropy

                PredictionsEnsemble = np.asarray(PredictionsEnsemble) # [5, 894, 256, 256]
                PredictionsEnsemble_By_Images = np.swapaxes(PredictionsEnsemble, 0, 1) # [894, 5, 256, 256]

                resolution = len(PredictionsEnsemble[0][0]) # 256

                # Multiprocessing ~~~ https://github.com/previtus/AttentionPipeline/blob/master/video_parser_v2/ThreadHerd.py
                predictions_N = len(PredictionsEnsemble[0])
                for prediction_i in range(predictions_N):
                    predictions = PredictionsEnsemble_By_Images[prediction_i] # 5 x 256x256

                    #start = timer()

                    entropy_image = None
                    sum_ent = 0

                    # incorrect calculation of entropy btw
                    ##entropy_image = np.apply_along_axis(arr=predictions, axis=0, func1d=entropy) # <<< The slow one ...
                    ##sum_ent = np.sum(entropy_image.flatten())

                    #end = timer()
                    #time = (end - start)
                    #print("Entropy "+str(time)+"s ("+str(time/60.0)+"min)") ## Entropy cca 0.011 min

                    #start = timer()

                    variance_image = np.var(predictions, axis=0)
                    sum_var = np.sum(variance_image.flatten())

                    #end = timer()
                    #time = (end - start)
                    #print("Variance "+str(time)+"s ("+str(time/60.0)+"min)") ## Variance cca 4.e-06

                    ##print("Sum entropy over whole image:", sum_ent) # min of this
                    #print("Sum variance over whole image:", sum_var) # max of this

                    do_viz = False
                    if do_viz:
                        fig = plt.figure(figsize=(10, 8))
                        for i in range(ModelEnsemble_N):
                            img = predictions[i]
                            ax = fig.add_subplot(1, ModelEnsemble_N+2, i+1)
                            plt.imshow(img, cmap='gray')
                            ax.title.set_text('Model '+str(i))

                        ax = fig.add_subplot(1, ModelEnsemble_N+2, ModelEnsemble_N+1)
                        plt.imshow(entropy_image, cmap='gray')
                        ax.title.set_text('Entropy Viz ('+str(sum_ent)+')')

                        ax = fig.add_subplot(1, ModelEnsemble_N+2, ModelEnsemble_N+2)
                        plt.imshow(variance_image, cmap='gray')
                        ax.title.set_text('Variance Viz ('+str(sum_var)+')')

                        plt.show()

                    entropies_over_samples.append(sum_ent)
                    variances_over_samples.append(sum_var)

                overall_indices  = np.append(overall_indices, remaining_indices)

                #if len(entropies_over_samples) > 120:
                #    break # few batches only

        #print("debug ,,,")
        #print("entropies_over_samples",entropies_over_samples)
        #print("variances_over_samples",variances_over_samples)
        #print("overall_indices",overall_indices)

        """
        #entropies_over_samples.sort() <- messes the rest
        #variances_over_samples.sort()
    
        plt.figure(figsize=(7, 7))  # w, h
        plt.plot(entropies_over_samples, 'red', label="entropies_over_samples")
        plt.plot(variances_over_samples, 'blue', label="variances_over_samples")
        plt.legend()
        #plt.ylim(0.0, 1.0)
        plt.show() #### plotting entropy over the set
        """

        if acquisition_function_mode == "Random":
            # note that in the case of Random, we don't need to calc the <variances_over_samples>
            selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(ITERATION_SAMPLE_SIZE)

            #print("random selected_indices", selected_indices)

        if acquisition_function_mode == "Ensemble":
            # indices 0 -> len(processed_remaining)
            # real corresponding indices are in remaining_indices

            Select_K = int( ITERATION_SAMPLE_SIZE - ENSEMBLE_tmp_how_many_from_random )
            semisorted_indices = np.argpartition(variances_over_samples, -Select_K) # < ints?
            selected_indices_A = semisorted_indices[-Select_K:]

            if ENSEMBLE_tmp_how_many_from_random == 0:
                selected_indices = selected_indices_A
            else:

                rest_indices = list(semisorted_indices[0:-Select_K])
                # Probably we want to get some of the images with highest variance ~ disagreement
                # and maybe we can also mix in some of the remaining images ...

                #print("ensemble selected_indices_A", len(selected_indices_A) ,"=>", selected_indices_A)

                from random import sample
                selected_indices_B = sample(rest_indices, ENSEMBLE_tmp_how_many_from_random) # random sampling without replacement

                #print("ensemble selected_indices_B", len(selected_indices_B),"=>",selected_indices_B)

                selected_indices = np.append(selected_indices_A,selected_indices_B)

            #print("ensemble (internal tmp) selected_indices", len(selected_indices),"=>",selected_indices)

            true_selected_indices = []
            for idx in selected_indices:
                true_selected_indices.append( overall_indices[idx] )

            #print("ensemble true_selected_indices", len(true_selected_indices),"=>",true_selected_indices)
            selected_indices = true_selected_indices

        # Visualize selected items? Maybe even with their scores

        packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
        TrainSet.add_items(packed_items)

        # Clean up

        for handler in ModelEnsemble:
            model = handler.model
            del model
            del handler

        del train_data
        del processed_train
        del processed_test
        #for i in range(3): gc.collect()
        keras.backend.clear_session()

        end = timer()
        time = (end - start)
        print("This iteration took "+str(time)+"s ("+str(time/60.0)+"min)")

        # Cheeky save per each iteration (so we can reconstruct old / broken plots)
        pixel_statistics = pixels_thresholds, xs_number_of_data, pixels_recalls, pixels_precisions, pixels_accuracies, pixels_f1s, Ns_changed, Ns_nochanged, pixels_AUCs
        tile_statistics = tiles_thresholds, xs_number_of_data, tiles_recalls, tiles_precisions, tiles_accuracies, tiles_f1s, Ns_changed, Ns_nochanged
        statistics = pixel_statistics, tile_statistics
        statistics = np.asarray(statistics)
        np.save("["+args.name+"]_al_statistics.npy", statistics)

    #- AL outline, have two sets - TrainSet and UnlabeledSet and move data in between them... (always everything we have?)
    #2b.) Acquisition function to select subset of the RemainingUnlabeledSet -> move it to the TrainSet

    pixel_statistics = pixels_thresholds, xs_number_of_data, pixels_recalls, pixels_precisions, pixels_accuracies, pixels_f1s, Ns_changed, Ns_nochanged, pixels_AUCs
    tile_statistics = tiles_thresholds, xs_number_of_data, tiles_recalls, tiles_precisions, tiles_accuracies, tiles_f1s, Ns_changed, Ns_nochanged
    statistics = pixel_statistics, tile_statistics
    statistics = np.asarray(statistics)
    np.save("["+args.name+"]_al_statistics.npy", statistics)
    #####statistics = np.load("al_statistics.npy")

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

    print("Ensemble tested on completely unbalanced!")
    print("Trying one with 1k to 40k balance and having it run properly - for few iterations at least -> later compare that between methods")






if __name__ == '__main__':
    args = parser.parse_args()

    start = timer()

    main(args)

    end = timer()
    time = (end - start)

    print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")

    import keras
    keras.backend.clear_session()
