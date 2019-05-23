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

def main(args):
    print(args)


    #import tensorflow as tf
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True
    #sess = tf.Session(config=config)

    """ TRAINING """
    epochs = int(args.model_epochs) # 50
    batch_size = int(args.model_batchsize) # 4
    augmentation = (args.train_augmentation == 'True') #False


    """ ACTIVE LEARNING """
    N_ITERATIONS = int(args.AL_iterations) #10
    # we have approx this ratio in our data 1072 : 81592

    INITIAL_SAMPLE_SIZE = int(args.AL_initialsample_size) #100
    TEST_SAMPLE_SIZE = int(args.AL_testsample_size) #250
    ITERATION_SAMPLE_SIZE = int(args.AL_iterationsample_size) #100


    acquisition_function_mode = args.AL_method #"Ensemble" / "Random"
    ModelEnsemble_N = int(args.AL_Ensemble_numofmodels)

    ENSEMBLE_tmp_how_many_from_random = 0 # not too many ... 1:4 seems like ok?


    # LOCAL OVERRIDES:

    acquisition_function_mode = "Random"
    ModelEnsemble_N = 1


    ## Loop starts with a small train set (INITIAL_SAMPLE_SIZE)
    ## then adds some number every iteration (ITERATION_SAMPLE_SIZE)
    ## also every iteration tests on a selected test sample

    import matplotlib.pyplot as plt
    import numpy as np


    #in_memory = False
    #in_memory = True # when it all fits its much faster
    #RemainingUnlabeledSet = get_balanced_dataset(in_memory)
    RemainingUnlabeledSet = get_unbalanced_dataset()

    # HAX
    print("-----------------------------")
    print("HAX: REMOVING SAMPLES SO WE DON'T GO MENTAL! (80k:1k ratio now 40k:1k ... still a big difference ...")
    REMOVE_SIZE = 40000
    #REMOVE_SIZE = 70000 # super HAX
    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset_balanced_classes(REMOVE_SIZE, 0.0)
    removed_items = RemainingUnlabeledSet.pop_items(selected_indices)

    RemainingUnlabeledSet.report()
    print("-----------------------------")


    settings = RemainingUnlabeledSet.settings
    TestSet = LargeDatasetHandler_AL(settings, "inmemory")
    #selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(TEST_SAMPLE_SIZE)
    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset_balanced_classes(TEST_SAMPLE_SIZE, 0.5) # should be balanced

    packed_items = RemainingUnlabeledSet.pop_items(selected_indices)

    TestSet.add_items(packed_items)
    print("Test set:")
    TestSet.report()
    N_change, N_nochange = TestSet.report_balance_of_class_labels()
    print("are we balanced in the test set?? Change:NoChange",N_change, N_nochange)

    TrainSet = LargeDatasetHandler_AL(settings, "inmemory")

    # Initial bump of data
    #selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(INITIAL_SAMPLE_SIZE)
    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset_balanced_classes(INITIAL_SAMPLE_SIZE, 0.5) # possibly also should be balanced?

    packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
    TrainSet.add_items(packed_items)

    # === Model and preprocessors

    test_data, test_paths, _ = TestSet.get_all_data_as_arrays()

    dataPreprocesser = DataPreprocesser_dataIndependent(settings, number_of_channels=4)
    trainTestHandler = TrainTestHandler(settings)
    evaluator = Evaluator(settings)

    # === Start looping:
    recalls = []
    precisions = []
    accuracies = []
    f1s = []
    xs_number_of_data = []

    thresholds = []

    Ns_changed = []
    Ns_nochanged = []

    #for active_learning_iteration in range(30): # 30*500 => 15k images
    for active_learning_iteration in range(N_ITERATIONS):
        xs_number_of_data.append(TrainSet.get_number_of_samples())

        # survived to 0-5 = 6*500 = 3000 images (I think that then the RAM was filled ...)
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

        print("Train shapes:", processed_train[0].shape, processed_train[1].shape, processed_train[2].shape)
        print("Test shapes:", processed_test[0].shape, processed_test[1].shape, processed_test[2].shape)

        # Init model
        #ModelHandler = ModelHandler_dataIndependent(settings, BACKBONE='resnet34')
        #model = ModelHandler.model

        ModelEnsemble = []
        Separate_Train_Eval__TrainAndSave = True # < we want to train new ones every iteration
        TMP__DontSave = False
        #Separate_Train_Eval__TrainAndSave = False # < this loads from the last interation (kept in case of mem errs)

        for i in range(ModelEnsemble_N):
            modelHandler = ModelHandler_dataIndependent(settings, BACKBONE='resnet34')
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

                _, broken_flag = trainTestHandler.train(ModelEnsemble[i].model, processed_train, epochs, batch_size, augmentation = augmentation, DEBUG_POSTPROCESSER=dataPreprocesser)

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



        print("Now I would test ...")

        DEBUG_SAVE_ALL_THR_PLOTS = "iteration_"+str(active_learning_iteration).zfill(2)+"_debugThrOverview"
        #DEBUG_SAVE_ALL_THR_PLOTS = None
        auto_thr = True # automatically choose thr which maximizes f1 score

        """ a
        from keras import backend as K
        from keras.models import load_model
    
        ModelHandler.save("tmp.h5")
    
        K.clear_session()
        K.set_learning_phase(1)
    
        ModelHandler2 = ModelHandler_dataIndependent(settings, BACKBONE='resnet34')
        ModelHandler2.load('tmp.h5')
        model2 = ModelHandler2.model
        """
        ##### trainTestHandler.test_STOCHASTIC_TESTS(model, processed_test, evaluator, postprocessor=dataPreprocesser, auto_thr=auto_thr, DEBUG_SAVE_ALL_THR_PLOTS=DEBUG_SAVE_ALL_THR_PLOTS)

        print("Selecting first model from the Ensemble for tests ...")
        model = ModelEnsemble[0].model

        recall, precision, accuracy, f1, threshold = trainTestHandler.test(model, processed_test, evaluator, postprocessor=dataPreprocesser, auto_thr=auto_thr, DEBUG_SAVE_ALL_THR_PLOTS=DEBUG_SAVE_ALL_THR_PLOTS)
        recalls.append(recall)
        precisions.append(precision)
        accuracies.append(accuracy)
        f1s.append(f1)
        thresholds.append(threshold)

        print("Now I would store/save the resulting model ...")


        print("Add new samples as per AL paradigm ... acquisition_function_mode=",acquisition_function_mode)

        # this is not necessary for Random ...
        if acquisition_function_mode == "Ensemble":

            entropies_over_samples = []
            variances_over_samples = []
            overall_indices = np.asarray([]) # important to concat these correctly across batches

            # BATCH THIS ENTIRE SEGMENT
            PER_BATCH = 2048 # Depends on memory really ...

            for batch in RemainingUnlabeledSet.generator_for_all_images(PER_BATCH, mode='dataonly'): # Yields a large batch sample
                remaining_indices = batch[0]
                remaining_data = batch[1]  # lefts and rights, no labels!
                print("MegaBatch (",len(entropies_over_samples),"/",RemainingUnlabeledSet.N_of_data,") = indices from id", remaining_indices[0], "to id", remaining_indices[-1])
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

                    print("Predicting for", nth, "model in the ensemble - for disagreement calculations")
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


        """ Experiment with the embedding of the model """
        """ Doesnt seem to work particularly well - dist is (prolly) not a good indicator for how much it changed
        remaining_data = None # np.ones((1, 10, 64, 64, 1)
    
        print("Layers >>> ")
        model.summary()
    
        from keras.models import Model
        from keras.layers.core import Lambda, Flatten
    
        #layer_name = 'model_2'
        #left_features = model.get_layer(layer_name).get_output_at(1)[0] # (?, 8, 8, 512) in resnet 34
        #right_features = model.get_layer(layer_name).get_output_at(2)[0] # (?, 8, 8, 512) in resnet 34
        concatenate_1 = model.get_layer("concatHighLvlFeat")
    
        def Crop(dim, start, end, **kwargs): #https://github.com/keras-team/keras/issues/890
            # Crops (or slices) a Tensor on a given dimension from start to end
            # example : to crop tensor x[:, :, 5:10]
            def func(x):
                dimension = dim
                if dimension == -1:
                    dimension = len(x.shape) - 1
                if dimension == 0:
                    return x[start:end]
                if dimension == 1:
                    return x[:, start:end]
                if dimension == 2:
                    return x[:, :, start:end]
                if dimension == 3:
                    return x[:, :, :, start:end]
                if dimension == 4:
                    return x[:, :, :, :, start:end]
            return Lambda(func, **kwargs)
    
        concat_feat_size = concatenate_1.output_shape[3]
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=[Flatten(name="left_feature")(Crop(3,0,int(concat_feat_size/2))(concatenate_1.output)),
                                                  Flatten(name="right_feature")(Crop(3,int(concat_feat_size/2),concat_feat_size)(concatenate_1.output))])
        #left_features = intermediate_layer_model.get_layer("left_feature")
        #right_features = intermediate_layer_model.get_layer("right_feature")
    
        for layer in intermediate_layer_model.layers:
            print(layer.output_shape)
    
        _, _, as_ones_and_zeros, idx_examples_bigger, idx_examples_smaller = TestSet.report_balance_of_class_labels(DEBUG_GET_IDX=True)
    
        train_L, train_R, train_V = processed_test # Just Trying / replace by batches from Unlabeled
        if train_L.shape[3] > 3:
            train_L = train_L[:, :, :, 1:4]
            train_R = train_R[:, :, :, 1:4]
    
        intermediate_output = intermediate_layer_model.predict([train_L, train_R], batch_size=32)
        left_features = intermediate_output[0] # N, 32768
        right_features = intermediate_output[1] # N, 32768
    
        print(intermediate_output)
        print(left_features.shape)
        print(right_features.shape)
    
        num_of_samples = len(left_features)
    
        def euclidean_distance(x, y):
            return np.sqrt(np.sum((x - y) ** 2))
    
        def cosine_similarity(x, y):
            return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
    
        distances = []
        distances_changed_gt = []
        distances_notchanged_gt = []
        for sample_i in range(num_of_samples):
            l = left_features[sample_i]
            r = right_features[sample_i]
    
            dist = cosine_similarity(l,r)
            #dist = euclidean_distance(l,r)
            distances.append(dist)
    
            if as_ones_and_zeros[sample_i]:
                distances_changed_gt.append(dist)
            else:
                distances_notchanged_gt.append(dist)
    
        #as_ones_and_zeros, idx_examples_bigger, idx_examples_smaller
    
    
        distances.sort()
        distances_changed_gt.sort()
        distances_notchanged_gt.sort()
    
        plt.figure(figsize=(7, 7))  # w, h
        plt.plot(distances, 'black', label="distances")
        plt.plot(distances_changed_gt, 'red', label="changed")
        plt.plot(distances_notchanged_gt, 'blue', label="not")
        plt.legend()
        #plt.ylim(0.0, 1.0)
        plt.show() #### What about a "good" model ? How does it behave there ??
        """
        """ End """


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
        statistics = thresholds, xs_number_of_data, recalls, precisions, accuracies, f1s, Ns_changed, Ns_nochanged
        statistics = np.asarray(statistics)
        np.save("al_statistics.npy", statistics)

    #- AL outline, have two sets - TrainSet and UnlabeledSet and move data in between them... (always everything we have?)
    #2b.) Acquisition function to select subset of the RemainingUnlabeledSet -> move it to the TrainSet

    statistics = thresholds, xs_number_of_data, recalls, precisions, accuracies, f1s, Ns_changed, Ns_nochanged
    statistics = np.asarray(statistics)
    np.save("al_statistics.npy", statistics)
    #####statistics = np.load("al_statistics.npy")

    thresholds, xs_number_of_data, recalls, precisions, accuracies, f1s, Ns_changed, Ns_nochanged = statistics

    print("thresholds:", thresholds)

    print("xs_number_of_data:", xs_number_of_data)
    print("recalls:", recalls)
    print("precisions:", precisions)
    print("accuracies:", accuracies)
    print("f1s:", f1s)

    print("Ns_changed:", Ns_changed)
    print("Ns_nochanged:", Ns_nochanged)

    plt.figure(figsize=(7, 7)) # w, h
    plt.plot(xs_number_of_data, thresholds, color='black', label="thresholds")

    plt.plot(xs_number_of_data, recalls, color='red', marker='o', label="recalls")
    plt.plot(xs_number_of_data, precisions, color='blue', marker='o', label="precisions")
    plt.plot(xs_number_of_data, accuracies, color='green', marker='o', label="accuracies")
    plt.plot(xs_number_of_data, f1s, color='orange', marker='o', label="f1s")

    #plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)

    plt.legend()
    plt.ylim(0.0, 1.0)

    plt.savefig("dbg_last_al_big_plot.png")
    plt.show()


    plt.figure(figsize=(7, 7)) # w, h

    N = len(Ns_changed)
    ind = np.arange(N)
    width = 0.35
    p1 = plt.bar(ind, Ns_changed, width)
    p2 = plt.bar(ind, Ns_nochanged, width, bottom=Ns_changed)

    plt.ylabel('number of data samples')
    plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)
    plt.legend((p1[0], p2[0]), ('Change', 'NoChange'))

    plt.savefig("dbg_last_al_balance_plot.png")
    plt.show()

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
