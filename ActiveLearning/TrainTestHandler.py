# FORCE CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import Debugger
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import sklearn.metrics

class TrainTestHandler(object):

    def __init__(self, settings):
        self.settings = settings
        self.debugger = Debugger.Debugger(settings)

    def train(self, model, training_set, epochs, batch, verbose = 1, augmentation = False, DEBUG_POSTPROCESSER = None):
        print("Training model:",model,"on training set (size", len(training_set[0]), "), Augment=",augmentation)

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

        if augmentation:

            from albumentations.core.transforms_interface import DualTransform
            class RandomRotate90x1(DualTransform):
                def apply(self, img, factor=0, **params):
                    return np.ascontiguousarray(np.rot90(img, factor))

                def get_params(self):
                    return {'factor': 1}

            class RandomRotate90x2(DualTransform):
                def apply(self, img, factor=0, **params):
                    return np.ascontiguousarray(np.rot90(img, factor))

                def get_params(self):
                    return {'factor': 2}

            class RandomRotate90x3(DualTransform):
                def apply(self, img, factor=0, **params):
                    return np.ascontiguousarray(np.rot90(img, factor))

                def get_params(self):
                    return {'factor': 3}

            from albumentations import (HorizontalFlip,
                                        VerticalFlip,
                                        Compose)
            from random import randint

            # using the help of https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb

            # we can grow the training set, and then shuffle it

            augmentations = []
            augmentations.append( RandomRotate90x1(p=1) ) # 90, 180 or 270 <- we need the same for the same of l=r=y
            augmentations.append( RandomRotate90x2(p=1) ) # 90, 180 or 270 <- we need the same for the same of l=r=y
            augmentations.append( RandomRotate90x3(p=1) ) # 90, 180 or 270 <- we need the same for the same of l=r=y
            augmentations.append( HorizontalFlip(p=1) )         # H reflection
            augmentations.append( VerticalFlip(p=1) )           # V reflection

            augmentations.append( Compose([VerticalFlip(p=1), RandomRotate90x1(p=1)]) ) # V reflection and then rotation
            augmentations.append( Compose([HorizontalFlip(p=1), RandomRotate90x1(p=1)]) ) # H reflection and then rotation

            # randomness inside the call breaks our case because we need to call it twice (two images and a mask)
            # perhaps temporary concat in channels could be a workaround?

            # include non-rigid transformations?
            #   Elastic def. = “Best Practices for Convolutional Neural Networks applied to Visual Document Analysis”
            #augmentations.append(ElasticTransformDET(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03))

            # more fancy ones ...
            # isn't deterministic for 3 images ....
            #original_height = original_width = self.dataset.datasetInstance.IMAGE_RESOLUTION
            #augmentations.append(RandomSizedCrop(p=1, min_max_height=(int(original_height/2.0), int(original_height)), height=original_height, width=original_width))

            # etc ...
            aug_lefts = []
            aug_rights = []
            aug_ys = []

            num_in_train = len(train_L)

            for i in range(num_in_train):
                #print(i)

                image_l = train_L[i]
                image_r = train_R[i]
                mask = train_V[i]

                if True:
                    # choose a random augmentation ... (we don't have mem for all of them!, solve by batches or smth ...)
                    aug_i = randint(0, len(augmentations)-1)
                    aug = augmentations[aug_i]
                    #for aug in augmentations:

                    augmented1 = aug(image=image_l, mask=mask)
                    augmented2 = aug(image=image_r, mask=mask)
                    aug_l = augmented1['image']
                    aug_y = augmented1['mask']
                    aug_lefts.append(np.asarray(aug_l))
                    aug_ys.append(np.asarray(aug_y))
                    aug_r = augmented2['image']
                    aug_rights.append(np.asarray(aug_r))
                    del aug_l
                    del aug_r
                    del aug_y
                    del augmented1
                    del augmented2

            if False:
                # for sake of showing:
                aug_lefts_tmp, aug_rights_tmp = DEBUG_POSTPROCESSER.postprocess_images(np.asarray(aug_lefts), np.asarray(aug_rights))
                print("aug_lefts_tmp.shape", aug_lefts_tmp.shape)
                print("aug_rights_tmp.shape", aug_rights_tmp.shape)

                #self.debugger.viewTripples(aug_lefts, aug_rights, aug_ys)
                by = 5
                off = 0
                while off < len(aug_lefts):
                    self.debugger.viewTripples(aug_lefts_tmp, aug_rights_tmp, aug_ys, how_many=by, off=off)
                    off += by

            aug_lefts = np.asarray(aug_lefts)
            aug_rights = np.asarray(aug_rights)
            aug_ys = np.asarray(aug_ys)

            if verbose > 2:
                print("aug_lefts.shape", aug_lefts.shape)
                print("aug_rights.shape", aug_rights.shape)
                print("aug_ys.shape", aug_ys.shape)

            # Adding them to the training set
            train_L = np.append(train_L, aug_lefts, axis=0)
            train_R = np.append(train_R, aug_rights, axis=0)
            train_V = np.append(train_V, aug_ys, axis=0)

            if verbose > 2:
                print("left images (aug train)")
                self.debugger.explore_set_stats(train_L)
                print("right images (aug train)")
                self.debugger.explore_set_stats(train_R)
                print("label images categorical (aug train)")
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

        broken_flag = False


        BIG_BATCHES = False
        # POSSIBLY MESSES IT UP:
        if BIG_BATCHES:
            # INSTEAD WITH BIG BATCHES ? - test if this can save us from GPU MEM problems
            for epoch in range(epochs):
                print("epoch %d" % epoch)

                big_batch_size = 500
                # we have len(train_V) data, we want to split it into batches ...
                start_i = 0
                total_N = len(train_V)
                while start_i < total_N:
                    end_i = start_i + big_batch_size
                    if end_i > total_N:
                        end_i = total_N

                    batch_train_L = train_L[start_i:end_i]
                    batch_train_R = train_R[start_i:end_i]
                    batch_train_V = train_V[start_i:end_i]
                    print("minibatch of", len(batch_train_L))

                    model.fit([batch_train_L, batch_train_R], batch_train_V, batch_size=batch, epochs=1, verbose=2)

                    start_i += big_batch_size

            history = None

        # try:
        #    for batch in TrainSet.generator_for_all_images(500, mode='datalabels'):  # Yields a large batch sample
        #        indices = batch[0]
        #        print("indices from", indices[0], "to", indices[-1])

        else:

            if True:
                #try:
                history = model.fit([train_L, train_R], train_V, batch_size=batch, epochs=epochs, verbose=2)
                #                         validation_data=([val_L, val_R], val_V), callbacks=callbacks

            """
            except Exception as e:
                print("Exception caught when trying to TRAIN:", e)
                print("Killing anyway ...")
                #assert False
                history = None
                broken_flag = True
            """

        return history, broken_flag


    def test_oldBack(self, model, testing_set, evaluator, postprocessor, auto_thr=False, DEBUG_SAVE_ALL_THR_PLOTS=None):
        print("Testing model:",model,"on test set (size", len(testing_set[0]),")")

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

        if auto_thr:
            best_thr, recall, precision, accuracy, f1 = evaluator.metrics_autothr_f1_max(predicted, test_V)
            print("Threshold automatically chosen as", best_thr)
            threshold = best_thr
        else:
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

        return recall, precision, accuracy, f1, threshold

    #####  tests of how BN behaves >>>> (might be del afterwards ...)

    def test_STOCHASTIC_TESTS(self, model, testing_set, evaluator, postprocessor, auto_thr=False, DEBUG_SAVE_ALL_THR_PLOTS=None):
        print("Testing model:",model,"on test set (size", len(testing_set[0]),")")

        test_L, test_R, test_V = testing_set

        if test_L.shape[3] > 3:
            # 3 channels only - rgb
            test_L = test_L[:,:,:,1:4]
            test_R = test_R[:,:,:,1:4]
        # label also reshape
        test_V_cat = to_categorical(test_V)

        """ description: http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
        In the case of BN, during training we use the mean and variance of the mini-batch to rescale the input. 
        On the other hand, during inference we use the moving average and variance that was estimated during training.
        """

        """ a is outside """

        """ b
        weights = model.get_weights()
        config = model.get_config()
        #model = Sequential.from_config(config) doesnt support multiinputoutu
        model = Model.from_config(config)
        model.set_weights(weights)

        loss = "categorical_crossentropy" # ~ should have been weighted_categorical_crossentropy
        metric = "categorical_accuracy"
        model.compile(optimizer=Adam(lr=0.00001), loss=loss, metrics=[metric, 'mse'])
        """


        # BatchNorm behavior in train/test may be according to this > https://github.com/keras-team/keras/issues/9522

        predicted1 = model.predict(x=[test_L, test_R], batch_size=4)
        predicted2 = model.predict(x=[test_L, test_R], batch_size=4)
        predicted3 = model.predict(x=[test_L, test_R], batch_size=8)
        predicted4 = model.predict(x=[test_L, test_R], batch_size=8)

        # different size of batch will change it as well!!! (with the train phase turned on)

        print(np.array_equal(predicted1, predicted2))
        print(np.array_equal(predicted1, predicted3))
        print(np.array_equal(predicted1, predicted4))

        print(np.array_equal(predicted3, predicted4)) #both 8 batch
        # in test time this remains the same
