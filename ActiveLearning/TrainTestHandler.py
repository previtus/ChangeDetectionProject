# FORCE CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import Debugger
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np

class TrainTestHandler(object):

    def __init__(self, settings):
        self.settings = settings
        self.debugger = Debugger.Debugger(settings)

    def train(self, model, training_set, epochs, batch, verbose = 1):
        print("Training model:",model,"on training set.")

        train_L, train_R, train_V = training_set

        # 3 channels only - rgb
        if train_L.shape[3] > 3:
            train_L = train_L[:,:,:,1:4]
            train_R = train_R[:,:,:,1:4]

        if verbose > 2:
            print("left images (train)")
            self.debugger.explore_set_stats(train_L)
            print("right images (train)")
            self.debugger.explore_set_stats(train_R)
            print("label images (train)")
            self.debugger.explore_set_stats(train_V)

        train_V = train_V.reshape(train_V.shape + (1,))

        # always softmax ...
        #self.use_sigmoid_or_softmax = 'softmax'
        #assert self.use_sigmoid_or_softmax == 'softmax'
        #if self.use_sigmoid_or_softmax == 'softmax':

        train_V = to_categorical(train_V)

        if verbose > 2:
            print("label images categorical (train)")
            self.debugger.explore_set_stats(train_V)

        history = model.fit([train_L, train_R], train_V, batch_size=batch, epochs=epochs, verbose=2)
        #                         validation_data=([val_L, val_R], val_V), callbacks=callbacks

        return history


    def test(self, model, testing_set, evaluator, postprocessor, verbose = 1, DEBUG_SAVE_ALL_THR_PLOTS=None):
        print("Testing model:",model,"on test set.")

        test_L, test_R, test_V = testing_set

        if test_L.shape[3] > 3:
            # 3 channels only - rgb
            test_L = test_L[:,:,:,1:4]
            test_R = test_R[:,:,:,1:4]
        # label also reshape
        test_V_cat = to_categorical(test_V)

        predicted = model.predict(x=[test_L, test_R], batch_size=4)
        #metrics = self.model.evaluate(x=[test_L, test_R], y=test_V_cat, verbose=0, batch_size=4)
        #metrics_info = self.model.metrics_names
        #print(list(zip(metrics_info, metrics)))

        # with just 2 classes I can hax:
        predicted = predicted[:, :, :, 1]

        predicted = postprocessor.postprocess_labels(predicted)

        threshold = 0.5 # 0.01 was too close
        # 0.01 and 0.1 was bad!
        recall, precision, accuracy, f1 = evaluator.calculate_recall_precision_accuracy(predicted, test_V, threshold=threshold, need_f1=True)

        # maybe call:
        if DEBUG_SAVE_ALL_THR_PLOTS is not None:
            show = False
            save = True
            name = DEBUG_SAVE_ALL_THR_PLOTS
            jump_by = 0.1 # 0.01
            evaluator.try_all_thresholds(predicted, test_V, np.arange(0.0, 1.0, jump_by),title_txt="Masks (all pixels 0/1) evaluated [Change Class]",
                                         show=show, save=save, name=name)

        return recall, precision, accuracy, f1
