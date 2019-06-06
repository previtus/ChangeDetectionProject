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


    import random
    import keras.backend as K
    import matplotlib.pyplot as plt
    import numpy as np

    T = 5
    batch_size = 16 # as it was when training

    train_data_indices = list(range(0,len(train_L)))

    f = K.function([model.layers[0].input, model.layers[1].input, K.learning_phase()],
                   [model.layers[-1].output])
    print("f", f)

    # For each sample?

    samples_N = 10
    predictions_for_sample = np.zeros((T,samples_N) + (256,256,)) # < T, SamplesN, 256x256 >

    for sample_id in range(samples_N):
        # like this it's probably slow ...

        sample = [test_L[sample_id], test_R[sample_id]]  # (2,256,256,3)
        sample = np.asarray(sample)

        for MC_iteration in range(T):
            selected_indices = random.sample(train_data_indices, batch_size-1)

            print("train_L[selected_indices] :: ", train_L[selected_indices].shape) # 15, 256,256,3
            print("sample :: ", sample.shape) # 2,256,256,3 ?

            train_sample = [ np.append(train_L[selected_indices], [sample[0]], 0),
                             np.append(train_R[selected_indices], [sample[1]], 0) ]
            train_sample = np.asarray(train_sample)

            print("MonteCarloBatchNormalization")
            print("T", T)
            print("batch_size", batch_size)
            print("sample.shape", sample.shape)
            print("train_sample.shape", train_sample.shape)

            # all in the training regime - local statistics get changed in each iteration
            predictions = f((np.asarray(train_sample[0], dtype=np.float32), np.asarray(train_sample[1], dtype=np.float32), 1))[0]
            print("predictions.shape", predictions.shape) # 16, 256,256,2

            sample_predicted = predictions[batch_size-1] # last one # 256,256,2

            sample_predicted = sample_predicted[:, :, 1]
            print("sample_predicted.shape", sample_predicted.shape) # 256,256

            predictions_for_sample[MC_iteration,sample_id, :,:] = sample_predicted

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
    import scipy

    for prediction_i in range(predictions_N):
        predictions = predictions_for_sample_By_Images[prediction_i]  # 5 x 256x256

        a_problematic_zone = np.finfo(float).eps # move 0-1 to 0.1 to 0.9
        helper_offset = np.ones_like(predictions)
        predictions = predictions * (1.0 - 2*a_problematic_zone) + helper_offset * (a_problematic_zone)

        def entropy_across_predictions(pixel_predictions):
            #print("pixel_predictions.shape", pixel_predictions.shape)
            T = len(pixel_predictions)
            p_sum = np.sum(pixel_predictions, axis=0)
            #assert len(pixel_predictions.shape) == 1

            pk0 = ( p_sum ) / T
            pk1 = 1 - ( p_sum ) / T

            entropy0 = - pk0 * np.log(pk0)
            entropy1 = - pk1 * np.log(pk1) # i think that this one can be ignored in two class case ... in theory ...

            """
            print("pk0", pk0)
            print("pk1", pk1)
            print("entropy0", entropy0)
            print("entropy1", entropy1)
            """
            return entropy0 + entropy1

        def ent_img_sumDiv(pixel_predictions):
            return np.sum(pixel_predictions, axis=0) / len(pixel_predictions)
        def ent_img_log(pk):
            return - pk * np.log(pk)

        startTMP = timer()
        # trying to write it faster !

        ent_img_pk0 = np.apply_along_axis(arr=predictions, axis=0, func1d=ent_img_sumDiv)
        ent_img_pk1 = np.ones_like(ent_img_pk0) - ent_img_pk0
        ent_img_ent0 = np.apply_along_axis(arr=ent_img_pk0, axis=0, func1d=ent_img_log)
        ent_img_ent1 = np.apply_along_axis(arr=ent_img_pk1, axis=0, func1d=ent_img_log)
        entropy_image = ent_img_ent0 + ent_img_ent1
        sum_ent = np.sum(entropy_image.flatten())

        endTMP = timer()
        timeTMP = (endTMP - startTMP)
        print("Entropy faster " + str(timeTMP) + "s (" + str(timeTMP / 60.0) + "min)")

        """
        Entropy faster 0.28297295499942265s (0.004716215916657044min)
        Entropy before 0.481015188008314s (0.008016919800138567min)

        startTMP = timer()

        entropy_image = np.apply_along_axis(arr=predictions, axis=0, func1d=entropy_across_predictions)
        sum_ent = np.sum(entropy_image.flatten())

        endTMP = timer()
        timeTMP = (endTMP - startTMP)
        print("Entropy before " + str(timeTMP) + "s (" + str(timeTMP / 60.0) + "min)")
        """

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

        startTMP = timer()

        bald_diff_image = np.apply_along_axis(arr=predictions, axis=0, func1d=BALD_diff)

        endTMP = timer()
        timeTMP = (endTMP - startTMP)
        print("Bald orig" + str(timeTMP) + "s (" + str(timeTMP / 60.0) + "min)")

        bald_image = -1 * ( entropy_image - bald_diff_image )
        sum_bald = np.sum(bald_image.flatten())

        variance_image = np.var(predictions, axis=0)
        sum_var = np.sum(variance_image.flatten())

        do_viz = True
        if do_viz:
            fig = plt.figure(figsize=(10, 8))
            for i in range(T):
                img = predictions[i]
                ax = fig.add_subplot(1, T + 3, i + 1)
                plt.imshow(img, cmap='gray', vmin = 0.0, vmax = 1.0)
                ax.title.set_text('Model ' + str(i))

            ax = fig.add_subplot(1, T + 3, T + 1)
            plt.imshow(entropy_image, cmap='gray', vmin = 0.0, vmax = 1.0)
            ax.title.set_text('Entropy (' + str(np.round(sum_ent,3)) + ')')

            #ax = fig.add_subplot(1, T + 3, T + 2)
            #plt.imshow(entropy_image_f, cmap='gray', vmin = 0.0, vmax = 1.0)
            #ax.title.set_text('Entropy_f (' + str(np.round(sum_ent_f,3)) + ')')

            ax = fig.add_subplot(1, T + 3, T + 2)
            plt.imshow(bald_image, cmap='gray') #, vmin = 0.0, vmax = 1.0)
            ax.title.set_text('BALD (' + str(np.round(sum_bald,3)) + ')')

            ax = fig.add_subplot(1, T + 3, T + 3)
            plt.imshow(variance_image, cmap='gray', vmin = 0.0, vmax = 1.0)
            ax.title.set_text('Variance (' + str(np.round(sum_var,3)) + ')')

            plt.show()



    # MCBN (sample, T, train_data, batch_size)
    # predictions_for_sample = []
    # for i in T:
    #   batch of train data <- random from train_data of size batch_size
    #   update_layer_statistics (= eval with training mode on)
    #   prediction = model.predict(sample)
    #   predictions.append(prediction)
    # return predictions

    nkhnkkjnjk

    # ----------------------------------------------------------
    # Predict data:

    print("about to predict data with", test_L.shape)
    predicted = model.model.model.predict(x=[test_L, test_R], batch_size=4)
    predicted = predicted[:, :, :, 1]
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
