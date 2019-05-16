# === Initialize sets - Unlabeled, Train and Test
import keras
from ActiveLearning.LargeDatasetHandler_AL import tmp_get_whole_dataset
from ActiveLearning.LargeDatasetHandler_AL import LargeDatasetHandler_AL
from ActiveLearning.ModelHandler_dataIndependent import ModelHandler_dataIndependent
from ActiveLearning.DataPreprocesser_dataIndependent import DataPreprocesser_dataIndependent
from ActiveLearning.TrainTestHandler import TrainTestHandler
from Evaluator import Evaluator
from timeit import default_timer as timer


#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)


import matplotlib.pyplot as plt
import numpy as np

TMP_WHOLE_UNBALANCED = True # Care!

in_memory = False
in_memory = True
"""
TempSetWithBalancedData = tmp_get_whole_dataset(in_memory)
RemainingUnlabeledSet = tmp_get_whole_dataset(in_memory, TMP_WHOLE_UNBALANCED)
settings = RemainingUnlabeledSet.settings

TestSet = LargeDatasetHandler_AL(settings, "inmemory")
#selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(750) # EXTREMELY unbalanced
#packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
selected_indices = TempSetWithBalancedData.sample_random_indice_subset(500)
packed_items = TempSetWithBalancedData.pop_items(selected_indices)
"""

RemainingUnlabeledSet = tmp_get_whole_dataset(in_memory)
settings = RemainingUnlabeledSet.settings
TestSet = LargeDatasetHandler_AL(settings, "inmemory")
selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(250) #250
packed_items = RemainingUnlabeledSet.pop_items(selected_indices)

TestSet.add_items(packed_items)
print("Test set:")
TestSet.report()
N_change, N_nochange = TestSet.report_balance_of_class_labels()
print("are we balanced in the test set?? Change:NoChange",N_change, N_nochange)

TrainSet = LargeDatasetHandler_AL(settings, "inmemory")

# === Model and preprocessors


test_data, test_paths = TestSet.get_all_data_as_arrays()

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
for active_learning_iteration in range(1):
    # survived to 0-5 = 6*500 = 3000 images (I think that then the RAM was filled ...)
    print("\n")
    print("==========================================================")
    print("\n")

    print("Active Learning iteration:", active_learning_iteration)
    start = timer()

    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(1000) # this can be 1:499 unbalanced!
    print(selected_indices)
    packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
    TrainSet.add_items(packed_items)

    print("Unlabeled:")
    RemainingUnlabeledSet.report()
    print("Train:")
    TrainSet.report()

    # what about the balance of the data?

    N_change, N_nochange = TrainSet.report_balance_of_class_labels()
    Ns_changed.append(N_change)
    Ns_nochanged.append(N_nochange)

    train_data, train_paths = TrainSet.get_all_data_as_arrays()
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
    ModelEnsemble_N = 5
    Separate_Train_Eval__TrainAndSave = True
    Separate_Train_Eval__TrainAndSave = False

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
        #epochs = 10 + active_learning_iteration # hmmmmmmm regularize?
        epochs = 50 # the baseline model had 100 epochs btw ...
        #epochs = 10
        ###epochs = 1 # debugs
        #batch = 16 # batch 16 for resnet50
        batch = 4 # 4 works nicely with the data for some reason ...
        #batch = 2 # so it survives as long as possible!

        augmentation = False
        #augmentation = True
        for i in range(ModelEnsemble_N):
            _, broken_flag = trainTestHandler.train(ModelEnsemble[i].model, processed_train, epochs, batch, augmentation = augmentation, DEBUG_POSTPROCESSER=dataPreprocesser)

        if broken_flag:
            print("Got as far as until AL iteration:", active_learning_iteration, " ... now breaking")
            break

        root = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/models_in_al/"
        # now save these models
        for i in range(ModelEnsemble_N):
            modelHandler = ModelEnsemble[i]
            modelHandler.save(root+"initTests_"+str(i))

    else:
        # Load and then eval! (check stuff first ...)
        root = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/models_in_al/"
        for i in range(ModelEnsemble_N):
            modelHandler = ModelEnsemble[i]
            modelHandler.load(root+"initTests_"+str(i))


    # have the models in ensemble make predictions
    # Function EnsemblePrediction() maybe even in batches?
    PredictionsEnsemble = []
    for handler in ModelEnsemble:
        model = handler.model

        test_L, test_R, test_V = processed_test

        if test_L.shape[3] > 3:
            # 3 channels only - rgb
            test_L = test_L[:,:,:,1:4]
            test_R = test_R[:,:,:,1:4]
        # label also reshape

        # only 5 images now!
        subs = 20
        test_L = test_L[0:subs]
        test_R = test_R[0:subs]

        from keras.utils import to_categorical
        test_V_cat = to_categorical(test_V)

        predicted = model.predict(x=[test_L, test_R], batch_size=4)
        predicted = predicted[:, :, :, 1] # 2 channels of softmax with 2 classes is useless - use just one

        PredictionsEnsemble.append(predicted)

    from scipy.stats import entropy

    predictions_N = len(PredictionsEnsemble[0])
    for prediction_i in range(predictions_N):
        #predictions_by_models = PredictionsEnsemble[:][prediction_i]
        predictions_by_models = []
        for model_i in range(ModelEnsemble_N):
            predictions_by_models.append( PredictionsEnsemble[model_i][prediction_i] )

        # (N, 256, 256)
        resolution = len(predictions_by_models[0])

        entropies = []
        variances = []

        entropy_image = np.zeros((resolution, resolution))
        variance_image = np.zeros((resolution, resolution))

        for pixel_x in range(resolution):
            for pixel_y in range(resolution):
                pixels = []
                for model_i in range(ModelEnsemble_N):
                    pixels.append(PredictionsEnsemble[model_i][prediction_i][pixel_x][pixel_y])

                if pixel_x < 10 and pixel_y < 2:
                    print("Pixels:", pixels)
                ent = entropy(pixels)
                var = np.var(pixels, 0)
                if pixel_x < 10 and pixel_y < 2:
                    print("entropy:", ent, "variance:", var)

                entropy_image[pixel_x][pixel_y] = ent
                variance_image[pixel_x][pixel_y] = var

                entropies.append(ent)
                variances.append(var)

        sum_ent = np.sum(entropies)
        sum_var = np.sum(variances)
        print("Sum entropy over whole image:", sum_ent) # min of this
        print("Sum variance over whole image:", sum_var) # max of this

        fig = plt.figure(figsize=(10, 8))
        for i in range(ModelEnsemble_N):
            img = predictions_by_models[i]
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


        print(np.asarray(predictions_by_models).shape)


    lkjkljjjl

    print("Now let's play with the ActiveLearning paradigm ...")

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


    recall, precision, accuracy, f1, threshold = trainTestHandler.test(model, processed_test, evaluator, postprocessor=dataPreprocesser, auto_thr=auto_thr, DEBUG_SAVE_ALL_THR_PLOTS=DEBUG_SAVE_ALL_THR_PLOTS)
    recalls.append(recall)
    precisions.append(precision)
    accuracies.append(accuracy)
    f1s.append(f1)
    xs_number_of_data.append(TrainSet.get_number_of_samples())
    thresholds.append(threshold)

    print("Now I would store/save the resulting model ...")

    del model
    del train_data
    del processed_train
    del processed_test
    #for i in range(3): gc.collect()
    keras.backend.clear_session() ### does this work???

    end = timer()
    time = (end - start)
    print("This iteration took "+str(time)+"s ("+str(time/60.0)+"min)")
    # time it took over iterations ...
    # 0 - 5 min
    # 1 - 7.5 min
    # ...
    # 5 - 17 min
    # 6 - 20 min



#- AL outline, have two sets - TrainSet and UnlabeledSet and move data in between them... (always everything we have?)
#2b.) Acquisition function to select subset of the RemainingUnlabeledSet -> move it to the TrainSet

print("thresholds:", thresholds)

print("xs_number_of_data:", xs_number_of_data)
print("recalls:", recalls)
print("precisions:", precisions)
print("accuracies:", accuracies)
print("f1s:", f1s)

print("Ns_changed:", Ns_changed)
print("Ns_nochanged:", Ns_nochanged)

plt.figure(figsize=(7, 7)) # w, h
plt.plot(xs_number_of_data, thresholds, 'black', label="thresholds")

plt.plot(xs_number_of_data, recalls, 'r', label="recalls")
plt.plot(xs_number_of_data, precisions, 'b', label="precisions")
plt.plot(xs_number_of_data, accuracies, 'g', label="accuracies")
plt.plot(xs_number_of_data, f1s, 'orange', label="f1s")

plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)

plt.legend()
plt.ylim(0.0, 1.0)

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

plt.show()



print("This one is from the full dataset - extremely messy and unbalanced!!!")



"""
eeeee

for e in range(3):
    print("epoch %d" % e)
    # mode > indices, dataonly, datalabels
    for batch in TrainSet.generator_for_all_images(500, mode='datalabels'): # Yields a large batch sample
        indices = batch[0]
        print("indices from", indices[0], "to", indices[-1])
"""