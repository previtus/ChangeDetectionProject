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

    def train(self, model, training_set, validation_set, epochs, batch, verbose = 1, augmentation = False, DEBUG_POSTPROCESSER = None, name="",
              FailSafeON=True, FailSafe__ValLossThr = 2.0 ):
        print("Training model:",model,"on training set (size", len(training_set[0]), "), Augment=",augmentation)

        train_L, train_R, train_V = training_set
        val_L, val_R, val_V = validation_set
        if val_L.shape[3] > 3:
            # 3 channels only - rgb
            val_L = val_L[:, :, :, 1:4]
            val_R = val_R[:, :, :, 1:4]

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
        val_V = val_V.reshape(val_V.shape + (1,))
        train_V = to_categorical(train_V)
        val_V = to_categorical(val_V)

        if verbose > 2:
            print("label images categorical (train)")
            self.debugger.explore_set_stats(train_V)

        broken_flag = False
        failed_training_flag = False

        history = model.fit([train_L, train_R], train_V, batch_size=batch, epochs=epochs, verbose=2,
                                         validation_data=([val_L, val_R], val_V) )#, callbacks=callbacks

        history.history["acc"] = history.history["categorical_accuracy"]  # we care about this one to show
        history.history["val_acc"] = history.history["val_categorical_accuracy"]

        print(history.history)
        self.debugger.nice_plot_history(history,added_plots = [], save=True, show=False, name=name+"_training")

        # Fail safe - failed_training_flag
        # if the last "val_acc" is too big
        # (optionally) if the recall (would be) 0.0 - might need additional checks though....
        if FailSafeON:
            val_losses = history.history["val_loss"]
            if val_losses[-1] > FailSafe__ValLossThr:
                print("Fail safe activated, the last model reached final VAL_LOSS=",val_losses[-1], "(which is >",FailSafe__ValLossThr,")!")
                failed_training_flag = True

        return history, broken_flag, failed_training_flag



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
