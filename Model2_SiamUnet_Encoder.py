# FORCE CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import Debugger, DataPreprocesser
import keras
from keras.layers import Input, Dense
from keras.models import Model
#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from random import randint

from keras.layers import Conv2D
from keras.layers import Activation
from keras.models import Model

from segmentation_models.utils import freeze_model
from segmentation_models.utils import legacy_support
from segmentation_models.backbones import get_backbone, get_feature_layers

from segmentation_models.unet.blocks import Transpose2D_block
from segmentation_models.utils import get_layer_number, to_tuple

from loss_weighted_crossentropy import weighted_categorical_crossentropy
from keras.callbacks import ModelCheckpoint

from Model2_builder import SiameseUnet

from albumentations import (PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma)

class Model2_SiamUnet_Encoder(object):
    """
    An intermediate between the code and bunch of tester models.
    """

    def __init__(self, settings, dataset):
        self.settings = settings
        self.dataset = dataset
        self.dataPreprocesser = dataset.dataPreprocesser
        self.debugger = Debugger.Debugger(settings)

        self.use_sigmoid_or_softmax = 'softmax'
        assert self.use_sigmoid_or_softmax == 'softmax'

        BACKBONE = 'resnet34'
        BACKBONE = 'resnet50' #batch 16
        BACKBONE = 'resnet101' #batch 8
        BACKBONE = 'seresnext50' #trying batch 16 as well
        custom_weights_file = "imagenet"

        #weights from imagenet finetuned on aerial data specific task - will it work? will it break?
        #custom_weights_file = "/scratch/ruzicka/python_projects_large/AerialNet_VariousTasks/model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep_dsm01proper.h5"

        resolution_of_input = self.dataset.datasetInstance.IMAGE_RESOLUTION
        self.model = self.create_model(backbone=BACKBONE, custom_weights_file=custom_weights_file, input_size = resolution_of_input, channels = 3)
        self.model.summary()

        self.local_setting_batch_size = 8 #32
        self.local_setting_epochs = 100 #100

        self.train_data_augmentation = True


        # saving paths for plots ...
        self.save_plot_path = "plots/"+self.settings.run_name+"_"

    def train(self, show=True, save=False):
        print("Train")

        train_L, train_R, train_V = self.dataset.train
        val_L, val_R, val_V = self.dataset.val

        # 3 channels only - rgb
        if train_L.shape[3] > 3:
            train_L = train_L[:,:,:,1:4]
            train_R = train_R[:,:,:,1:4]
            val_L = val_L[:,:,:,1:4]
            val_R = val_R[:,:,:,1:4]

        # label also reshape

        print("left images (train)")
        self.debugger.explore_set_stats(train_L)
        print("right images (train)")
        self.debugger.explore_set_stats(train_R)
        print("label images (train)")
        self.debugger.explore_set_stats(train_V)

        added_plots = []

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

        if self.train_data_augmentation:
            # using the help of https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb

            # we can grow the training set, and then shuffle it
            train_L
            train_R
            train_V

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
                    aug_lefts_tmp, aug_rights_tmp = self.dataPreprocesser.postprocess_images(np.asarray(aug_lefts), np.asarray(aug_rights))

                    #self.debugger.viewTripples(aug_lefts, aug_rights, aug_ys)
                    by = 5
                    off = i * by
                    while off < len(aug_lefts):
                        self.debugger.viewTripples(aug_lefts_tmp, aug_rights_tmp, aug_ys, how_many=by, off=off)
                        off += by

            aug_lefts = np.asarray(aug_lefts)
            aug_rights = np.asarray(aug_rights)
            aug_ys = np.asarray(aug_ys)

            print("aug_lefts.shape", aug_lefts.shape)
            print("aug_rights.shape", aug_rights.shape)
            print("aug_ys.shape", aug_ys.shape)

            # Adding them to the training set
            train_L = np.append(train_L, aug_lefts, axis=0)
            train_R = np.append(train_R, aug_rights, axis=0)
            train_V = np.append(train_V, aug_ys, axis=0)

            print("left images (aug train)")
            self.debugger.explore_set_stats(train_L)
            print("right images (aug train)")
            self.debugger.explore_set_stats(train_R)
            print("label images categorical (aug train)")
            self.debugger.explore_set_stats(train_V)



        # don't do this before Augmentation
        train_V = train_V.reshape(train_V.shape + (1,))
        val_V = val_V.reshape(val_V.shape + (1,))

        if self.use_sigmoid_or_softmax == 'softmax':
            train_V = to_categorical(train_V)
            val_V = to_categorical(val_V)

        print("label images categorical (train)")
        self.debugger.explore_set_stats(train_V)

        checkpoint = ModelCheckpoint("model2best_so_far_for_eastly_stops.h5", monitor='val_categorical_accuracy',
                                     verbose=1, save_best_only=True, mode='max')
        callbacks = [checkpoint]  # should we prefer the best model instead?? maybe no, the val is pretty small
        # callbacks = []

        # Regular training:
        history = self.model.fit([train_L, train_R], train_V, batch_size=self.local_setting_batch_size,
                                 epochs=self.local_setting_epochs,
                                 validation_data=([val_L, val_R], val_V), verbose=2,
                                 callbacks=callbacks)  # 2 ~ 1 line each ep
        # print(history.history)

        if self.use_sigmoid_or_softmax == 'sigmoid':
            history.history["acc"] = history.history["binary_accuracy"]  # we care about this one to show
            history.history["val_acc"] = history.history["val_binary_accuracy"]
            added_plots = []
        else:
            history.history["acc"] = history.history["categorical_accuracy"]  # we care about this one to show
            history.history["val_acc"] = history.history["val_categorical_accuracy"]
            added_plots = []

        print(history.history)


        self.debugger.nice_plot_history(history,added_plots, save=save, show=show, name=self.save_plot_path+"training")

    def save(self, path=""):
        if path == "":
            self.model.save_weights(self.settings.large_file_folder+"last_trained_model_weights.h5")
        else:
            self.model.save_weights(path)
        print("Saved model weights.")

    def load(self, path=""):
        if path == "":
            self.model.load_weights(self.settings.large_file_folder+"last_trained_model_weights.h5")
        else:
            self.model.load_weights(path)
        print("Loaded model weights.")

    def test(self, evaluator, show = True, save = False):
        print("Test")

        test_L, test_R, test_V = self.dataset.test

        if test_L.shape[3] > 3:
            # 3 channels only - rgb
            test_L = test_L[:,:,:,1:4]
            test_R = test_R[:,:,:,1:4]
        # label also reshape
        if self.use_sigmoid_or_softmax == 'softmax':
            test_V_cat = to_categorical(test_V)
        else:
            test_V_cat = test_V.reshape(test_V.shape + (1,))

        #predicted = self.model.predict(x=[test_L, test_R])
        #metrics = self.model.evaluate(x=[test_L, test_R], y=test_V_cat, verbose=0)
        predicted = self.model.predict(x=[test_L, test_R], batch_size=4)
        metrics = self.model.evaluate(x=[test_L, test_R], y=test_V_cat, verbose=0, batch_size=4)
        metrics_info = self.model.metrics_names
        print(list(zip(metrics_info, metrics)))

        if self.use_sigmoid_or_softmax == 'softmax':
            # with just 2 classes I can hax:
            predicted = predicted[:, :, :, 1]
            # else:
            #predicted = np.argmax(predicted, axis=3)
            #print("predicted.shape:", predicted.shape)

        else:
            # chop off that last dimension
            predicted = predicted.reshape(predicted.shape[:-1])

        # undo preprocessing steps?
        predicted = self.dataPreprocesser.postprocess_labels(predicted)

        #print("MASK EVALUATION")
        #print("trying thresholds ...")
        
        evaluator.try_all_thresholds(predicted, test_V, np.arange(0.0,1.0,0.05), title_txt="Masks (all pixels 0/1) evaluated [Change Class]", show=show,save=save, name=self.save_plot_path)

        # Evaluator
        #evaluator.histogram_of_predictions(predicted)
        print("threshold=0.5")
        evaluator.calculate_metrics(predicted, test_V, threshold=0.5)

        test_L, test_R = self.dataPreprocesser.postprocess_images(test_L, test_R)

        if test_L.shape[3] > 3:
            # 3 channels only - rgb
            test_L = test_L[:,:,:,1:4]
            test_R = test_R[:,:,:,1:4]


        print("left images (test)")
        self.debugger.explore_set_stats(test_L)
        print("right images (test)")
        self.debugger.explore_set_stats(test_R)
        print("label images (test)")
        self.debugger.explore_set_stats(test_V)
        print("predicted images (test)")
        self.debugger.explore_set_stats(predicted)

        if show:
            off = 0
            by = 4
            by = min(by, len(test_L))
            while off < len(predicted):
                #self.debugger.viewTripples(test_L, test_R, test_V, how_many=4, off=off)
                self.debugger.viewQuadrupples(test_L, test_R, test_V, predicted, how_many=by, off=off, show=show,save=save)
                off += by
        if save:
            off = 0
            by = 4
            by = min(by, len(test_L))
            until_n = min(by*8, len(test_L))
            while off < until_n:
                #self.debugger.viewTripples(test_L, test_R, test_V, how_many=4, off=off)
                self.debugger.viewQuadrupples(test_L, test_R, test_V, predicted, how_many=by, off=off, show=show,save=save, name=self.save_plot_path+"quad"+str(off))
                off += by


    def test_show_on_train_data_to_see_overfit(self, evaluator):
        print("Debug, showing performance on Train data!")


    def create_model(self, backbone='resnet34', custom_weights_file = "imagenet", input_size = 112, channels = 3):

        model = SiameseUnet(backbone, encoder_weights=custom_weights_file, classes=2, activation='softmax',
                            input_shape=(input_size, input_size, channels), encoder_freeze=False)
        print("Model loaded:")
        print("model.input", model.input)
        print("model.output", model.output)

        # Loss and metrics:

        loss = "categorical_crossentropy"

        weights = [1, 3]
        loss = weighted_categorical_crossentropy(weights)

        metric = "categorical_accuracy"
        model.compile(optimizer=Adam(lr=0.00001), loss=loss, metrics=[metric, 'mse'])

        return model

