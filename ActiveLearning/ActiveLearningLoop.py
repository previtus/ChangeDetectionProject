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
parser.add_argument('-seed', help='random seed (for multiple runs)', default="30")

parser.add_argument('-model_epochs', help='How many epochs will each model train?', default="100") #50? 100?
parser.add_argument('-model_batchsize', help='How big batch size for each model? (This is limited by the GPUs memory)', default="16")
parser.add_argument('-model_backbone', help='Encoder', default="resnet34")

parser.add_argument('-train_augmentation', help='Turn on augmentation? (one new image for each image in the train set - effectively doubling the dataset size)', default="False")

parser.add_argument('-AL_iterations', help='Number of iterations in the Active Learning loop', default="10")
parser.add_argument('-AL_initialsample_size', help='Start with this many sampled images in the training set', default="50")
parser.add_argument('-AL_testsample_size', help='Have this many balanced sample images in the testing set (used for plots)', default="200")
parser.add_argument('-AL_valsample_size', help='Have this many balanced sample images in the validation set (used for automatic thr choice and val errs)', default="200")
parser.add_argument('-AL_iterationsample_size', help='Add this many images in each iteration', default="100")

parser.add_argument('-AL_method', help='Sampling method (choose from "Random", "Ensemble", "MonteCarloBatchNormalization")', default="Ensemble")

parser.add_argument('-AL_AcquisitionFunction', help='For any method other than Random (choose from "Variance", "Entropy", "BALD")', default="Variance")

parser.add_argument('-AL_Ensemble_numofmodels', help='If we chose Ensemble, how many models are there?', default="3")
parser.add_argument('-AL_MCBN_numofruns', help='If we chose Ensemble, how many models are there?', default="4")

parser.add_argument('-DEBUG_remove_from_dataset', help='Debug to remove random samples without change from the original dataset...', default="40000")
parser.add_argument('-DEBUG_loadLastALModels', help='Debug function - load last saved model weights instead of training ...', default="False")


parser.add_argument('-resume', help='Resume from a folder (target the ones made in iterations) ...', default="")

import random
import numpy
import pickle

def save_random_states(path):
    random_state = random.getstate()
    nprandom_state = numpy.random.get_state()
    with open(path, 'wb') as f:
        pickle.dump([random_state,nprandom_state], f)

def load_random_states(path):
    f = open(path, "rb")
    bin_data = f.read()
    random_state, nprandom_state = pickle.loads(bin_data)
    random.setstate(random_state)
    numpy.random.set_state(nprandom_state)


def main(args):
    args_txt = str(args)
    print(args_txt)

    #seed_num = 80 #50 #30 was done
    seed_num = int(args.seed)

    args_name = args.name


    # keras and it's training is not influenced by this

    #import tensorflow as tf
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True
    #sess = tf.Session(config=config)

    """ RESUME """
    resume = args.resume

    """ TRAINING """
    epochs = int(args.model_epochs) # 50
    batch_size = int(args.model_batchsize) # 4
    augmentation = (args.train_augmentation == 'True') #False
    model_backbone = args.model_backbone


    """ ACTIVE LEARNING """
    N_ITERATIONS = int(args.AL_iterations) #10
    # we have approx this ratio in our data 1072 : 81592

    INITIAL_SAMPLE_SIZE = int(args.AL_initialsample_size) #100
    TEST_SAMPLE_SIZE = int(args.AL_testsample_size) #200
    VAL_SAMPLE_SIZE = int(args.AL_valsample_size) #200
    ITERATION_SAMPLE_SIZE = int(args.AL_iterationsample_size) #100


    acquisition_function_mode = args.AL_method #"Ensemble" / "Random" / ""
    acquisition_function = args.AL_AcquisitionFunction #"Variance" / "Entropy" / ""
    ModelEnsemble_N = int(args.AL_Ensemble_numofmodels)

    if acquisition_function_mode == "Random":
        ModelEnsemble_N = 1
    if acquisition_function_mode == "MonteCarloBatchNormalization":
        ModelEnsemble_N = 1

    MCBN_T = int(args.AL_MCBN_numofruns)

    ENSEMBLE_tmp_how_many_from_random = 0 # Hybridization


    REMOVE_SIZE = int(args.DEBUG_remove_from_dataset)
    DEBUG_loadLastALModels = (args.DEBUG_loadLastALModels == 'True')
    DEBUG_skip_evaluation = False
    DEBUG_skip_loading_most_of_data_batches = False # Doesnt do what it should otherwise ...
    # SHOULD REMAIN HARDCODED:
    DEBUG_CLEANMEM_FOR_20_MODELS = True

    threshold_fineness = 0.05 # default

    # New feature, failsafe for models in Ensembles ...
    FailSafeON = True # default
    FailSafe__ValLossThr = 1.5 # default # maybe should be 1.5 ??? defo should be 1.5


    # LOCAL OVERRIDES:
    """
    acquisition_function_mode = "Ensemble"
    ModelEnsemble_N = 1
    INITIAL_SAMPLE_SIZE = 100
    N_ITERATIONS = 2
    epochs = 35
    REMOVE_SIZE = 75000
    #DEBUG_loadLastALModels = True
    #DEBUG_skip_evaluation = True # won't save the plots, but jumps directly to the AL acquisition functions
    
    threshold_fineness = 0.1 # 0.05 makes nicer plots

    ###acquisition_function_mode = "Ensemble"
    ###acquisition_function_mode = "MonteCarloBatchNormalization" # <<< Work in progress ...
    """


    random.seed(seed_num) # samples
    numpy.random.seed(seed_num) # shuffles


    ## Loop starts with a small train set (INITIAL_SAMPLE_SIZE)
    ## then adds some number every iteration (ITERATION_SAMPLE_SIZE)
    ## also every iteration tests on a selected test sample

    import matplotlib.pyplot as plt
    import numpy as np

    if resume is not "":
        assert resume[-1] == "/"

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
    N_change_val, N_nochange_val = ValSet.report_balance_of_class_labels()
    print("are we balanced in the val set?? Change:NoChange",N_change_val, N_nochange_val)



    TrainSet = LargeDatasetHandler_AL(settings, "inmemory")

    # Initial bump of data
    #selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(INITIAL_SAMPLE_SIZE)
    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset_balanced_classes(INITIAL_SAMPLE_SIZE, 0.5) # possibly also should be balanced?

    packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
    TrainSet.add_items(packed_items)



    RemainingUnlabeledSet.settings.tmp_marker = str(seed_num) + acquisition_function_mode[0:3] + acquisition_function[0:3] + "_" + str(MCBN_T) + "_" + str(ModelEnsemble_N)
    TestSet.settings.tmp_marker = str(seed_num) + acquisition_function_mode[0:3] + acquisition_function[0:3] + "_" + str(MCBN_T) + "_" + str(ModelEnsemble_N)
    ValSet.settings.tmp_marker = str(seed_num) + acquisition_function_mode[0:3] + acquisition_function[0:3] + "_" + str(MCBN_T) + "_" + str(ModelEnsemble_N)
    TrainSet.settings.tmp_marker = str(seed_num) + acquisition_function_mode[0:3] + acquisition_function[0:3] + "_" + str(MCBN_T) + "_" + str(ModelEnsemble_N)


    # === Model and preprocessors

    test_data, test_paths, _ = TestSet.get_all_data_as_arrays()
    val_data, val_paths, _ = ValSet.get_all_data_as_arrays()

    dataPreprocesser = DataPreprocesser_dataIndependent(settings, number_of_channels=4)
    trainTestHandler = TrainTestHandler(settings)
    evaluator = Evaluator(settings)
    evaluator.settings.tmp_marker = str(seed_num) + acquisition_function_mode[0:3] + acquisition_function[0:3] + "_" + str(MCBN_T) + "_" + str(ModelEnsemble_N)
    settings.tmp_marker = str(seed_num) + acquisition_function_mode[0:3] + acquisition_function[0:3] + "_" + str(MCBN_T) + "_" + str(ModelEnsemble_N)

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

    IterationRange = list(range(N_ITERATIONS))

    # LOAD:
    if resume is not "":
        print("Resuming run from: ", resume)

        # load settings - and apply them where they cause changes!
        statistics_resume = resume+"resume_statistics.npy"
        all_settings_resume = resume+"resume_all_settings.npy"
        all_sets_resume = resume+"resume_all_sets.npy"
        randomstates_resume = resume+"resume_randomstates.pickle"

        # statistics
        statistics = np.load(statistics_resume)
        pixel_statistics, tile_statistics = statistics
        pixels_thresholds, xs_number_of_data, pixels_recalls, pixels_precisions, pixels_accuracies, pixels_f1s, Ns_changed, Ns_nochanged, pixels_AUCs = pixel_statistics
        tiles_thresholds, xs_number_of_data, tiles_recalls, tiles_precisions, tiles_accuracies, tiles_f1s, Ns_changed, Ns_nochanged = tile_statistics

        # settings
        all_settings = np.load(all_settings_resume)
        active_learning_iteration, seed_num, args_name, resume, epochs, augmentation, model_backbone, N_ITERATIONS, \
                    INITIAL_SAMPLE_SIZE, TEST_SAMPLE_SIZE, VAL_SAMPLE_SIZE, ITERATION_SAMPLE_SIZE, \
                    acquisition_function_mode, acquisition_function, ModelEnsemble_N, MCBN_T, \
                    ENSEMBLE_tmp_how_many_from_random, DEBUG_loadLastALModels, DEBUG_skip_evaluation, \
                    DEBUG_skip_loading_most_of_data_batches, threshold_fineness, FailSafeON, FailSafe__ValLossThr = all_settings

        # Types ...
        active_learning_iteration = int(active_learning_iteration)
        seed_num = int(seed_num)
        epochs = int(epochs)
        N_ITERATIONS = int(N_ITERATIONS)
        INITIAL_SAMPLE_SIZE = int(INITIAL_SAMPLE_SIZE)
        TEST_SAMPLE_SIZE = int(TEST_SAMPLE_SIZE)
        VAL_SAMPLE_SIZE = int(VAL_SAMPLE_SIZE)
        ITERATION_SAMPLE_SIZE = int(ITERATION_SAMPLE_SIZE)
        ModelEnsemble_N = int(ModelEnsemble_N)
        MCBN_T = int(MCBN_T)
        ENSEMBLE_tmp_how_many_from_random = int(ENSEMBLE_tmp_how_many_from_random)
        # this may cause "is not" to not work as expected btw ...
        # loads as  <class 'numpy.str_'> -> into str()
        acquisition_function_mode = str(acquisition_function_mode)
        acquisition_function = str(acquisition_function)

        augmentation = (augmentation == "True")
        DEBUG_loadLastALModels = (DEBUG_loadLastALModels == "True")
        DEBUG_skip_evaluation = (DEBUG_skip_evaluation == "True")
        DEBUG_skip_loading_most_of_data_batches = (DEBUG_skip_loading_most_of_data_batches == "True")
        FailSafeON = (FailSafeON == "True")

        threshold_fineness = float(threshold_fineness)
        FailSafe__ValLossThr = float(FailSafe__ValLossThr)

        done_iterations = active_learning_iteration
        print("done_iterations", done_iterations, type(done_iterations))
        print("FailSafeON", FailSafeON, type(FailSafeON))
        IterationRange = list(range((done_iterations + 1), N_ITERATIONS))
        print("Resuming from having finished iteration ", done_iterations, " remaining:", IterationRange)

        evaluator.settings.tmp_marker = str(seed_num) + acquisition_function_mode[0:3] + acquisition_function[0:3] + "_" + str(MCBN_T) + "_" + str(ModelEnsemble_N)
        settings.tmp_marker = str(seed_num) + acquisition_function_mode[0:3] + acquisition_function[0:3] + "_" + str(MCBN_T) + "_" + str(ModelEnsemble_N)

        # rebuild the sets ..............................................................................
        set_indices = np.load(all_sets_resume)
        TrainSetIndices, ValSetIndices, RemainingSetIndices, TestSetIndices = set_indices

        WholeUnlabeledSet = get_unbalanced_dataset()

        settings = WholeUnlabeledSet.settings
        TestSet = LargeDatasetHandler_AL(settings, "inmemory")
        ValSet = LargeDatasetHandler_AL(settings, "inmemory")
        TrainSet = LargeDatasetHandler_AL(settings, "inmemory")
        RemainingUnlabeledSet = LargeDatasetHandler_AL(settings, "ondemand")
        RemainingUnlabeledSet.original_indices = WholeUnlabeledSet.original_indices
        RemainingUnlabeledSet.N_of_data = WholeUnlabeledSet.N_of_data
        RemainingUnlabeledSet.per_tile_class = WholeUnlabeledSet.per_tile_class
        RemainingUnlabeledSet.has_per_tile_class_computed = WholeUnlabeledSet.has_per_tile_class_computed
        RemainingUnlabeledSet.dataaug_descriptors = WholeUnlabeledSet.dataaug_descriptors
        RemainingUnlabeledSet.paths = WholeUnlabeledSet.paths
        RemainingUnlabeledSet.settings = WholeUnlabeledSet.settings

        trainset_packed_items = WholeUnlabeledSet.pop_items(TrainSetIndices)
        TrainSet.add_items(trainset_packed_items)
        testset_packed_items = WholeUnlabeledSet.pop_items(TestSetIndices)
        TestSet.add_items(testset_packed_items)
        valset_packed_items = WholeUnlabeledSet.pop_items(ValSetIndices)
        ValSet.add_items(valset_packed_items)
        remainingset_packed_items = WholeUnlabeledSet.pop_items(RemainingSetIndices)
        RemainingUnlabeledSet.add_items(remainingset_packed_items)

        load_random_states(randomstates_resume)
        print("sample random -->", numpy.random.rand(1, 6))

        print("====================================== Reports:")

        print("RemainingUnlabeledSet set:")
        RemainingUnlabeledSet.report()
        print("RemainingUnlabeledSet.N_of_data = ", RemainingUnlabeledSet.N_of_data)

        print("Train set:")
        TrainSet.report()
        #N_change_test, N_nochange_test = TrainSet.report_balance_of_class_labels()
        print("are we balanced in the train set?? Change:NoChange", N_change_test, N_nochange_test)

        print("Test set:")
        TestSet.report()
        #N_change_test, N_nochange_test = TestSet.report_balance_of_class_labels()
        print("are we balanced in the test set?? Change:NoChange", N_change_test, N_nochange_test)

        print("Validation set:")
        ValSet.report()
        #N_change_val, N_nochange_val = TestSet.report_balance_of_class_labels()
        print("are we balanced in the val set?? Change:NoChange", N_change_val, N_nochange_val)

        print("======================================")

        test_data, test_paths, _ = TestSet.get_all_data_as_arrays()
        val_data, val_paths, _ = ValSet.get_all_data_as_arrays()

        dataPreprocesser = DataPreprocesser_dataIndependent(settings, number_of_channels=4)
        trainTestHandler = TrainTestHandler(settings)
        evaluator = Evaluator(settings)

        print("----- loaded")


    #for active_learning_iteration in range(30): # 30*500 => 15k images
    for active_learning_iteration in IterationRange:
        xs_number_of_data.append(TrainSet.get_number_of_samples())
        # survived to 0-5 = 6*500 = 3000 images (I think that then the RAM was filled ...)

        # Save args log:
        DEBUG_SAVE_ALL_THR_FOLDER = "["+args_name+"]_iteration_"+str(active_learning_iteration).zfill(2)+"_debugThrOverview/"
        DEBUG_SAVE_ALL_THR_PLOTS = DEBUG_SAVE_ALL_THR_FOLDER+"iteration_"+str(active_learning_iteration).zfill(2)
        #DEBUG_SAVE_ALL_THR_PLOTS = None
        if not os.path.exists(DEBUG_SAVE_ALL_THR_FOLDER):
            os.makedirs(DEBUG_SAVE_ALL_THR_FOLDER)
        file = open(DEBUG_SAVE_ALL_THR_FOLDER+"args.txt", "w")
        file.write(args_txt)
        file.close()

        print("\n")
        print("==========================================================")
        print("Folder >>> ", DEBUG_SAVE_ALL_THR_PLOTS)
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


        if Separate_Train_Eval__TrainAndSave:
            # === Train!:
            print("Now I would train ...")
            for i in range(ModelEnsemble_N):
                print("Training model",i,"in the ensemble (out of",ModelEnsemble_N,"):")

                AllowedFailsafeRetrains = 10 # Not to actually get stuck
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
                    else:
                        failed_training_flag = False

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

        import keras.backend as K
        from keras.utils import to_categorical

        if acquisition_function_mode == "MonteCarloBatchNormalization":
            print("Started with the MCBN preparation - first we save the current model (to later reset to it for each MCBN iteration)... into > AL_model_original_for_MCBN" + settings.tmp_marker + ".h5")
            ModelEnsemble[0].model.save(settings.large_file_folder+"AL_model_original_for_MCBN" + settings.tmp_marker + ".h5")
            #make N copies of the model - train each a bit away with its BN's

            train_L, train_R, train_V = processed_val
            if train_L.shape[3] > 3:
                # 3 channels only - rgb
                train_L = train_L[:, :, :, 1:4]
                train_R = train_R[:, :, :, 1:4]
            train_V = train_V.reshape(train_V.shape + (1,))
            train_V = to_categorical(train_V)

            batch_size_mctrain = batch_size  # as it was when training - default was 4, will that be ok?

            train_data_indices = list(range(0, len(train_L)))

            if DEBUG_CLEANMEM_FOR_20_MODELS:
                keras.backend.clear_session() # the model is saved, no longer needed in mem

            for MC_iteration in range(MCBN_T):
                model_new_h = ModelHandler_dataIndependent(settings, BACKBONE=model_backbone)
                model_new_h.load(settings.large_file_folder+"AL_model_original_for_MCBN" + settings.tmp_marker + ".h5")
                model = model_new_h.model

                selected_indices = random.sample(train_data_indices, batch_size_mctrain * 4)

                print("train_L[selected_indices] :: ", train_L[selected_indices].shape)  # 16, 256,256,3

                train_sample = [train_L[selected_indices], train_R[selected_indices]]
                train_sample = np.asarray(train_sample)
                train_sample_labels = np.asarray(train_V[selected_indices])

                print("MonteCarloBatchNormalization")
                print("MCBN_T", MCBN_T)
                print("batch_size_mctrain", batch_size_mctrain)
                print("train_sample.shape", train_sample.shape)

                # freeze everything besides BN layers
                for i, layer in enumerate(model.layers[2].layers):
                    name = layer.name
                    if "bn" not in name:
                        # freeeze layer which is not BN:
                        layer.trainable = False
                for i, layer in enumerate(model.layers):
                    name = layer.name
                    if "bn" not in name:
                        # freeeze layer which is not BN:
                        layer.trainable = False

                print("Separately training an MCBN model", MC_iteration, " into > "+"AL_model_original_for_MCBN_T" + str(MC_iteration).zfill(2) + settings.tmp_marker + ".h5")
                model.fit(x=[train_sample[0], train_sample[1]], y=train_sample_labels, batch_size=batch_size_mctrain, epochs=25, verbose=2)

                # ModelEnsemble.append(model_new_h)
                model_new_h.save(settings.large_file_folder+"AL_model_original_for_MCBN_T" + str(MC_iteration).zfill(2) + settings.tmp_marker + ".h5")
                del model_new_h

                # With lots of models this also explodes - will have to Keras Clean
                if DEBUG_CLEANMEM_FOR_20_MODELS:
                    keras.backend.clear_session() # the model is saved, no longer needed in mem


        # this is not necessary for Random ...
        if acquisition_function_mode == "Ensemble" or acquisition_function_mode == "MonteCarloBatchNormalization":


            BALD_over_samples = []
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

                # PS: these batches can be pretty small ... I wonder if I shouldn't accumulate them until the wanted [PER_BATCH]
                # but how would I know how many would come in the next batch?

                #if Accumulated + len(processed_remaining) < (PER_BATCH / 2):
                #    Accumulator.append(processed_remaining)

                PredictionsEnsemble = []
                if acquisition_function_mode == "Ensemble":
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

                # TODO: {idea} For MonteCarloBatchNormalization it might be better to make the models as copies of themselves to <ModelEnsemble> and train them before these batches come!

                elif acquisition_function_mode == "MonteCarloBatchNormalization":
                    print(" ==== Work In Progress ==== ")

                    ##############################################################################################
                    ##############################################################################################
                    ##############################################################################################

                    test_L, test_R = processed_remaining

                    if test_L.shape[3] > 3:
                        # 3 channels only - rgb
                        test_L = test_L[:, :, :, 1:4]
                        test_R = test_R[:, :, :, 1:4]

                    batch_size_mctest = 4  # as it was when training

                    # For each sample?
                    samples_N_BIGBatch = len(test_L)
                    res = 256
                    predictions_for_sample = np.zeros((MCBN_T, samples_N_BIGBatch) + (res, res,))  # < T, SamplesN, 256x256 >
                    print("(init) predictions_for_sample.shape", predictions_for_sample.shape)

                    # FOR MINIBATCH IN BIGBATCH ...
                    minibatch_size = batch_size_mctest
                    times_minibatch = int(samples_N_BIGBatch/minibatch_size)
                    minibatch_i = 0

                    for MC_iteration in range(MCBN_T):
                        # ! IN EACH BATCH WE LOAD
                        # ! BUT AFTER WE ALSO CLEAN

                        model_h = ModelHandler_dataIndependent(settings, BACKBONE=model_backbone)
                        model = model_h.model
                        # model_h.load(settings.large_file_folder + "AL_model_original_for_MCBN.h5") # does this work? YES
                        # What if we change the model exactly?
                        for i, layer in enumerate(model.layers[2].layers):
                            name = layer.name
                            if "bn" not in name:
                                # freeeze layer which is not BN:
                                layer.trainable = False
                        for i, layer in enumerate(model.layers):
                            name = layer.name
                            if "bn" not in name:
                                # freeeze layer which is not BN:
                                layer.trainable = False
                        # now the models are the same ...
                        model_h.load(
                            settings.large_file_folder + "AL_model_original_for_MCBN_T" + str(MC_iteration).zfill(
                                2) + settings.tmp_marker + ".h5")
                        model = model_h.model
                        f = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()],
                                       [model.layers[
                                            -1].output])  # potentially wasting memory over time? this probably adds tf items

                        minibatch_i = 0
                        while minibatch_i < samples_N_BIGBatch: # ps: this is the slowpoke - predicting 4 samples per loop
                            a = minibatch_i
                            b = min(minibatch_i + minibatch_size, samples_N_BIGBatch)
                            minibatch_i += minibatch_size

                            sample = [test_L[a:b], test_R[a:b]]  # (32, 2,256,256,3)
                            sample = np.asarray(sample)

                            print("(mini)sample.shape", sample.shape)

                            predicted = \
                                f((np.asarray(sample[0], dtype=np.float32), np.asarray(sample[1], dtype=np.float32), 1))[0]

                            predicted = predicted[:, :, :, 1]
                            print("(mini)predicted.shape", sample.shape)

                            predictions_for_sample[MC_iteration, a:b, :, :] = predicted
                            print("predictions_for_sample.shape", predictions_for_sample.shape) # (5, 175, 256, 256) # this fills 4 samples per iteration until cca 1049 (+- batch prop)

                            del sample
                            del predicted

                        print("got until (end of range) b=", b)

                        keras.backend.clear_session() # necessary cleanup for GPU mem

                    PredictionsEnsemble = predictions_for_sample
                    ##############################################################################################
                    # end if MCBN ################################################################################
                    ##############################################################################################
                    del model
                    del model_h
                    del f
                    del predictions_for_sample


                # This is still inside this one batch!
                PredictionsEnsemble = np.asarray(PredictionsEnsemble) # [5, 894, 256, 256]
                PredictionsEnsemble_By_Images = np.swapaxes(PredictionsEnsemble, 0, 1) # [894, 5, 256, 256]

                print("PredictionsEnsemble_By_Images.shape", PredictionsEnsemble_By_Images.shape) # PredictionsEnsemble_By_Images.shape (5, 1, 175, 256, 256)

                resolution = len(PredictionsEnsemble[0][0]) # 256

                # Multiprocessing ~~~ https://github.com/previtus/AttentionPipeline/blob/master/video_parser_v2/ThreadHerd.py
                predictions_N = len(PredictionsEnsemble[0])
                for prediction_i in range(predictions_N):
                    predictions = PredictionsEnsemble_By_Images[prediction_i] # 5 x 256x256

                    # No need
                    # a_problematic_zone = 0.0001 # move 0-1 to 0.1 to 0.9
                    # helper_offset = np.ones_like(predictions)
                    # predictions = predictions * (1.0 - 2*a_problematic_zone) + helper_offset * (a_problematic_zone)

                    sum_bald = 0
                    sum_ent = 0

                    if acquisition_function is not "Variance": # "Variance", "Entropy", "BALD")
                    
                        #BALD calc 1.467359378002584s (0.0244559896333764min)

                        def BALD_diff(pixel_predictions):
                            # Bayesian Active Learning by Disagreement = BALD = https://arxiv.org/abs/1112.5745
                            #T = len(pixel_predictions)
                            #assert len(pixel_predictions.shape) == 1

                            accum = 0
                            for val in pixel_predictions:
                                #if val == 0.0:
                                #    val += np.finfo(float).eps
                                #elif val == 1.0:
                                #    val -= np.finfo(float).eps

                                accum0 = - val * np.log(val)
                                accum1 = - (1-val) * np.log(1-val)
                                accum += accum0 + accum1

                            return accum

                        def ent_img_sumDiv(pixel_predictions):
                            return np.sum(pixel_predictions, axis=0) / len(pixel_predictions)
                        def ent_img_log(pk):
                            return - pk * np.log(pk)

                        startTMP = timer()

                        #Entropy calc 0.44229524800903164s (0.007371587466817194min)

                        ent_img_pk0 = np.apply_along_axis(arr=predictions, axis=0, func1d=ent_img_sumDiv)
                        ent_img_pk1 = np.ones_like(ent_img_pk0) - ent_img_pk0
                        ent_img_ent0 = np.apply_along_axis(arr=ent_img_pk0, axis=0, func1d=ent_img_log)
                        ent_img_ent1 = np.apply_along_axis(arr=ent_img_pk1, axis=0, func1d=ent_img_log)
                        entropy_image = ent_img_ent0 + ent_img_ent1
                        sum_ent = np.sum(entropy_image.flatten())

                        endTMP = timer()
                        timeTMP = (endTMP - startTMP)
                        print("Entropy calc " + str(timeTMP ) + "s (" + str(timeTMP  / 60.0) + "min)")

                        startTMP = timer()

                        bald_diff_image = np.apply_along_axis(arr=predictions, axis=0, func1d=BALD_diff)

                        bald_image = -1 * ( entropy_image - bald_diff_image )
                        sum_bald = np.sum(bald_image.flatten())

                        endTMP = timer()
                        timeTMP = (endTMP - startTMP)
                        print("BALD calc " + str(timeTMP ) + "s (" + str(timeTMP / 60.0) + "min)")


                    #startTMP = timer()

                    #Variance calc 0.00033402000553905964s (5.56700009231766e-06min)
                    variance_image = np.var(predictions, axis=0)
                    sum_var = np.sum(variance_image.flatten())

                    #endTMP = timer()
                    #timeTMP = (endTMP - startTMP)
                    #print("Variance calc " + str(timeTMP ) + "s (" + str(timeTMP / 60.0) + "min)")


                    do_viz = False
                    if do_viz:
                        #if prediction_i < 32: # see first few !
                        fig = plt.figure(figsize=(10, 8))
                        for i in range(len(PredictionsEnsemble)):
                            img = predictions[i]
                            ax = fig.add_subplot(1, len(PredictionsEnsemble) + 3, i + 1)
                            plt.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
                            ax.title.set_text('Model ' + str(i))

                        """
                        ax = fig.add_subplot(1, len(PredictionsEnsemble) + 3, ModelEnsemble_N + 1)
                        plt.imshow(entropy_image, cmap='gray', vmin=0.0, vmax=1.0)
                        ax.title.set_text('Entropy (' + str(np.round(sum_ent, 3)) + ')')

                        ax = fig.add_subplot(1, len(PredictionsEnsemble) + 3, ModelEnsemble_N + 2)
                        plt.imshow(bald_image, cmap='gray', vmin=0.0, vmax=1.0)
                        ax.title.set_text('BALD (' + str(np.round(sum_bald, 3)) + ')')
                        """

                        ax = fig.add_subplot(1, len(PredictionsEnsemble) + 3, ModelEnsemble_N + 3)
                        plt.imshow(variance_image, cmap='gray', vmin=0.0, vmax=1.0)
                        ax.title.set_text('Variance (' + str(np.round(sum_var, 3)) + ')')

                        plt.show()

                    BALD_over_samples.append(sum_bald)
                    entropies_over_samples.append(sum_ent)
                    variances_over_samples.append(sum_var)

                overall_indices  = np.append(overall_indices, remaining_indices)

                if DEBUG_skip_loading_most_of_data_batches:
                    if len(entropies_over_samples) > 120:
                        break # few batches only

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

        if acquisition_function_mode == "Ensemble" or acquisition_function_mode == "MonteCarloBatchNormalization":
            # indices 0 -> len(processed_remaining)
            # real corresponding indices are in remaining_indices

            used_values = variances_over_samples
            ### parser.add_argument('-AL_AcquisitionFunction', help='For any method other than Random (choose from "Variance", "Entropy", "BALD")', default="Variance")
            if acquisition_function == "Entropy":
                used_values = entropies_over_samples
            elif acquisition_function == "BALD":
                used_values = BALD_over_samples

            Select_K = int( ITERATION_SAMPLE_SIZE - ENSEMBLE_tmp_how_many_from_random )
            semisorted_indices = np.argpartition(used_values, -Select_K) # < ints?
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
        np.save("["+args_name+"]_al_statistics.npy", statistics) # also, for easier checking
        np.save(DEBUG_SAVE_ALL_THR_FOLDER+"resume_statistics.npy", statistics)


        # SAVE THIS ITERATION
        print("Saving this iteration for resume into folder: ", DEBUG_SAVE_ALL_THR_FOLDER)
        # Save all settings + iteration N
        all_settings = []
        all_settings = active_learning_iteration, seed_num, args_name, resume, epochs, augmentation, model_backbone, N_ITERATIONS, \
                       INITIAL_SAMPLE_SIZE, TEST_SAMPLE_SIZE, VAL_SAMPLE_SIZE, ITERATION_SAMPLE_SIZE, \
                       acquisition_function_mode, acquisition_function, ModelEnsemble_N, MCBN_T, \
                       ENSEMBLE_tmp_how_many_from_random,DEBUG_loadLastALModels,DEBUG_skip_evaluation, \
                       DEBUG_skip_loading_most_of_data_batches, threshold_fineness, FailSafeON, FailSafe__ValLossThr
        all_settings = np.asarray(all_settings)
        np.save(DEBUG_SAVE_ALL_THR_FOLDER+"resume_all_settings.npy", all_settings)

        # Save Train + Val + Remaining sets
        TrainSetIndices = TrainSet.get_all_indices_for_saving()
        ValSetIndices = ValSet.get_all_indices_for_saving()
        TestSetIndices = TestSet.get_all_indices_for_saving()
        RemainingSetIndices = RemainingUnlabeledSet.get_all_indices_for_saving()
        set_indices = TrainSetIndices, ValSetIndices, RemainingSetIndices, TestSetIndices
        set_indices = np.asarray(set_indices)
        np.save(DEBUG_SAVE_ALL_THR_FOLDER+"resume_all_sets.npy", set_indices)

        # random states (not all that important as we expect stochasticity in the models anyway...)
        save_random_states(DEBUG_SAVE_ALL_THR_FOLDER+"resume_randomstates.pickle")
        print("sample random -->", numpy.random.rand(1, 6))

        print("----- saved")

        print("=====<this was saved>================================= Reports:")

        print("RemainingUnlabeledSet set:")
        RemainingUnlabeledSet.report()

        print("Train set:")
        TrainSet.report()
        #N_change_test, N_nochange_test = TrainSet.report_balance_of_class_labels()
        print("are we balanced in the train set?? Change:NoChange", N_change_test, N_nochange_test)

        print("Test set:")
        TestSet.report()
        #N_change_test, N_nochange_test = TestSet.report_balance_of_class_labels()
        print("are we balanced in the test set?? Change:NoChange", N_change_test, N_nochange_test)

        print("Validation set:")
        ValSet.report()
        #N_change_val, N_nochange_val = TestSet.report_balance_of_class_labels()
        print("are we balanced in the val set?? Change:NoChange", N_change_val, N_nochange_val)

        print("=====<this was saved>=================================")

    #- AL outline, have two sets - TrainSet and UnlabeledSet and move data in between them... (always everything we have?)
    #2b.) Acquisition function to select subset of the RemainingUnlabeledSet -> move it to the TrainSet

    pixel_statistics = pixels_thresholds, xs_number_of_data, pixels_recalls, pixels_precisions, pixels_accuracies, pixels_f1s, Ns_changed, Ns_nochanged, pixels_AUCs
    tile_statistics = tiles_thresholds, xs_number_of_data, tiles_recalls, tiles_precisions, tiles_accuracies, tiles_f1s, Ns_changed, Ns_nochanged
    statistics = pixel_statistics, tile_statistics
    statistics = np.asarray(statistics)
    np.save("["+args_name+"]_al_statistics.npy", statistics)
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

    plt.savefig("["+args_name+"]_dbg_last_al_big_plot_pixelsScores.png")
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

    plt.savefig("["+args_name+"]_dbg_last_al_big_plot_tilesScores.png")
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

    plt.savefig("["+args_name+"]_dbg_last_al_balance_plot.png")
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
