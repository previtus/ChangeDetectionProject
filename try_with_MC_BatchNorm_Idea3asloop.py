# FORCE CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load a nice model
# /scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_100ep_ImagenetWgenetW_resnet50-16batch_Augmentation1to1_ClassWeights1to3_TestVal_[KFold_0z5].h5

one_model_path = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_100ep_ImagenetWgenetW_resnet50-16batch_Augmentation1to1_ClassWeights1to3_TestVal_[KFold_0z5].h5"

# Run prediction on a val or test set with different Monte Carlo runs

import matplotlib, os
#
#matplotlib.use("Agg")

import Dataset, Settings, ModelHandler, Evaluator
from timeit import default_timer as timer
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Project: Change detection on aerial images.')
parser.add_argument('-one_model_path', help='Path to h5 file.', default=one_model_path)
parser.add_argument('-name', help='run name - will output in this dir', default="tmp")
parser.add_argument('-model_backend', help='Model used in the encoder part of the U-Net structures model', default="resnet50")
parser.add_argument('-train_epochs', help='How many epochs', default='100')
parser.add_argument('-train_batch', help='How big batch size', default='16')

def main(args):
    import keras.backend as K

    print(args)

    settings = Settings.Settings(args)
    settings.TestDataset_Fold_Index = 0
    settings.TestDataset_K_Folds = 5
    settings.model_backend = args.model_backend
    settings.train_batch = args.train_batch
    settings.train_epochs = args.train_epochs

    dataset = Dataset.Dataset(settings)
    evaluator = Evaluator.Evaluator(settings)

    show = False
    save = True

    model_h = ModelHandler.ModelHandler(settings, dataset)
    model_h.model.load(args.one_model_path)
    model = model_h.model.model

    """ # One could also reload all the weights manually ...
    # care about a model inside a model!
    weights_list = []
    for i, layer in enumerate(model.layers[3:]):
        weights_list.append(layer.get_weights())
    for i, layer in enumerate(model.layers[3:]):
        weights = weights_list[i]
        name = layer.name
        print(name, len(weights), len(layer.weights))
        # restore by:
        if "_bn" in name:
            layer.set_weights(weights) # Batch normalization weights are: [gamma, beta, mean, std]
    """

    # data prep:
    test_set_processed = dataset.dataPreprocesser.apply_on_a_set_nondestructively(dataset.test)
    train_set_processed = dataset.dataPreprocesser.apply_on_a_set_nondestructively(dataset.train)
    test_L, test_R, test_V = test_set_processed
    train_L, train_R, train_V = train_set_processed

    if test_L.shape[3] > 3:
        # 3 channels only - rgb
        test_L = test_L[:, :, :, 1:4]
        test_R = test_R[:, :, :, 1:4]
        train_L = train_L[:, :, :, 1:4]
        train_R = train_R[:, :, :, 1:4]


    train_V = train_V.reshape(train_V.shape + (1,))
    from keras.utils import to_categorical
    train_V = to_categorical(train_V)

    import random
    import matplotlib.pyplot as plt
    import numpy as np

    T = 5
    batch_size = 16 # as it was when training

    train_data_indices = list(range(0,len(train_L)))

    f = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()],
                   [model.layers[-1].output])
    print("f", f)

    # For each sample?

    samples_N = 32
    predictions_for_sample = np.zeros((T,samples_N) + (256,256,)) # < T, SamplesN, 256x256 >
    sample = [test_L[0:samples_N], test_R[0:samples_N]]  # (16, 2,256,256,3)
    sample = np.asarray(sample)

    for MC_iteration in range(T):
        selected_indices = random.sample(train_data_indices, batch_size*4)

        print("train_L[selected_indices] :: ", train_L[selected_indices].shape)  # 16, 256,256,3
        print("sample :: ", sample.shape)  # 16, 2,256,256,3 ?

        train_sample = [train_L[selected_indices], train_R[selected_indices]]
        train_sample = np.asarray(train_sample)
        train_sample_labels = np.asarray(train_V[selected_indices])

        print("MonteCarloBatchNormalization")
        print("T", T)
        print("batch_size", batch_size)
        print("sample.shape", sample.shape)
        print("train_sample.shape", train_sample.shape)

        """
        # complete revert? Arguably not necessary
        model_h = ModelHandler.ModelHandler(settings, dataset) # < this will be slow
        model_h.model.load(args.one_model_path)
        model = model_h.model.model
        #model.load_weights(args.one_model_path) # revert at each MC_iteration start
        """

        # freeze everything besides BN layers
        for i, layer in enumerate(model.layers[2].layers):
            name = layer.name
            if "bn" not in name:
                # freeeze layer which is not BN:
                layer.trainable = False
            #print(name, layer.trainable)
        for i, layer in enumerate(model.layers):
            name = layer.name
            if "bn" not in name:
                # freeeze layer which is not BN:
                layer.trainable = False
                # else layer.stateful = True ?
            #print(name, layer.trainable)

        """ Without it shouts a warning, but seems alright
        # Re-Compile! (after changing the trainable param.)
        from keras.optimizers import Adam
        from loss_weighted_crossentropy import weighted_categorical_crossentropy
        loss = "categorical_crossentropy"
        weights = [1, 3]
        loss = weighted_categorical_crossentropy(weights)
        metric = "categorical_accuracy"
        model.compile(optimizer=Adam(lr=0.00001), loss=loss, metrics=[metric, 'mse'])
        #
        """

        model.fit(x=[train_sample[0], train_sample[1]], y=train_sample_labels, batch_size=16, epochs=25, verbose=2)

        """ # revert weights? (another way instead of loading from the .h5 file)
        weights_list = []
        for i, layer in enumerate(model.layers[3:]):
            weights_list.append(layer.get_weights())

        model.load_weights(args.one_model_path) # revert

        for i, layer in enumerate(model.layers[3:]):
            weights = weights_list[i]

            name = layer.name
            print(name, len(weights), len(layer.weights))

            if "_bn" in name:
                layer.set_weights(weights) # Batch normalization weights are: [gamma, beta, mean, std]
        """


        # model.predict would be nice to be able to batch easily
        # .... however ... predictions = model.predict(x=[sample[0], sample[1]], batch_size=16, verbose=2) # q: can i replace the f(...) with this?
        # it's not behaving

        ## don't want to make a new function every time though...
        #X#f = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()],
        #X#               [model.layers[-1].output])

        predictions = \
        f((np.asarray(sample[0], dtype=np.float32), np.asarray(sample[1], dtype=np.float32), 1))[0]

        # here BNs use exponentially weighted (/running) avg of the params for each layer from values it has seen during training
        # (sort of like the latest average value)
        # Ps: second prediction here is the same

        print("predictions.shape", predictions.shape)  # 16, 256,256,2

        sample_predicted = predictions[:, :, :, 1]
        print("sample_predicted.shape", sample_predicted.shape)  # 256,256

        predictions_for_sample[MC_iteration, :, :, :] = sample_predicted

    #print("are they equal? 0-1", np.array_equal(predictions_for_sample[0], predictions_for_sample[1]))
    #print("are they equal? 1-2", np.array_equal(predictions_for_sample[1], predictions_for_sample[2]))
    #print("are they equal? 2-3", np.array_equal(predictions_for_sample[2], predictions_for_sample[3]))

    predictions_for_sample = np.asarray(predictions_for_sample)  # [5, 100, 256, 256]

    print("predictions_for_sample ::", predictions_for_sample.shape)
    predictions_for_sample_By_Images = np.swapaxes(predictions_for_sample, 0, 1)  # [100, 5, 256, 256]

    print("predictions_for_sample_By_Images ::", predictions_for_sample_By_Images.shape)

    resolution = len(predictions_for_sample[0][0])  # 256
    predictions_N = len(predictions_for_sample[0])

    print("predictions_N:", predictions_N)

    for prediction_i in range(predictions_N):
        predictions = predictions_for_sample_By_Images[prediction_i]  # 5 x 256x256

        variance_image = np.var(predictions, axis=0)
        sum_var = np.sum(variance_image.flatten())

        do_viz = True
        if do_viz:
            fig = plt.figure(figsize=(10, 8))
            for i in range(T):
                img = predictions[i]
                ax = fig.add_subplot(1, T + 1, i + 1)
                plt.imshow(img, cmap='gray')
                ax.title.set_text('Model ' + str(i))

            ax = fig.add_subplot(1, T + 1, T + 1)
            plt.imshow(variance_image, cmap='gray')
            ax.title.set_text('Variance Viz (' + str(sum_var) + ')')

            plt.show()


    # MCBN (sample, T, train_data, batch_size)
    # predictions_for_sample = []
    # for i in T:
    #   batch of train data <- random from train_data of size batch_size
    #   update_layer_statistics (= eval with training mode on)
    #   prediction = model.predict(sample)
    #   predictions.append(prediction)
    # return predictions

    nkhnkkjnjkghhhhhh

    # ----------------------------------------------------------

if __name__ == '__main__':
    args = parser.parse_args()

    start = timer()

    print("### EVALUATION OF LOADED TRAINED MODEL ###")
    main(args)

    end = timer()
    time = (end - start)

    print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")

    import keras
    keras.backend.clear_session()
