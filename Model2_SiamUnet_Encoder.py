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

        resolution_of_input = self.dataset.datasetInstance.IMAGE_RESOLUTION
        self.model = self.create_model(backbone=BACKBONE, input_size = resolution_of_input, channels = 3)
        self.model.summary()

        self.local_setting_batch_size = 32 #32
        self.local_setting_epochs = 10 #100

        self.train_data_augmentation = False # todo possibly ...


    def train(self):
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
        train_V = train_V.reshape(train_V.shape + (1,))
        val_V = val_V.reshape(val_V.shape + (1,))

        print("left images (train)")
        self.debugger.explore_set_stats(train_L)
        print("right images (train)")
        self.debugger.explore_set_stats(train_R)
        print("label images (train)")
        self.debugger.explore_set_stats(train_V)

        if self.use_sigmoid_or_softmax == 'softmax':
            train_V = to_categorical(train_V)
            val_V = to_categorical(val_V)

        added_plots = []

        if self.train_data_augmentation:
            print("FOOO")

        else:
            checkpoint = ModelCheckpoint("model2best_so_far_for_eastly_stops.h5", monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
            callbacks = [checkpoint]
            callbacks = []

            # Regular training:
            history = self.model.fit([train_L, train_R], train_V, batch_size=self.local_setting_batch_size, epochs=self.local_setting_epochs,
                                     validation_data=([val_L, val_R], val_V), verbose=2, callbacks=callbacks) # 2 ~ 1 line each ep
            #print(history.history)

            if self.use_sigmoid_or_softmax == 'sigmoid':
                history.history["acc"] = history.history["binary_accuracy"]  # we care about this one to show
                history.history["val_acc"] = history.history["val_binary_accuracy"]
                added_plots = []
            else:
                history.history["acc"] = history.history["categorical_accuracy"]  # we care about this one to show
                history.history["val_acc"] = history.history["val_categorical_accuracy"]
                added_plots = []

            print(history.history)

        self.debugger.nice_plot_history(history,added_plots)

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

    def test(self, evaluator):
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

        predicted = self.model.predict(x=[test_L, test_R])
        metrics = self.model.evaluate(x=[test_L, test_R], y=test_V_cat, verbose=0)
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
        evaluator.try_all_thresholds(predicted, test_V, np.arange(0.0,1.0,0.01), title_txt="Masks (all pixels 0/1) evaluated [Change Class]")

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

        off = 0
        while off < len(predicted):
            #self.debugger.viewTripples(test_L, test_R, test_V, how_many=4, off=off)
            self.debugger.viewQuadrupples(test_L, test_R, test_V, predicted, how_many=4, off=off)
            off += 4

    def test_show_on_train_data_to_see_overfit(self, evaluator):
        print("Debug, showing performance on Train data!")


    def create_model(self, backbone='resnet34', input_size = 112, channels = 3):

        #custom_weights_file = "/scratch/ruzicka/python_projects_large/AerialNet_VariousTasks/model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep_dsm01proper.h5"
        custom_weights_file = "imagenet"

        model = SiameseUnet(backbone, encoder_weights=custom_weights_file, classes=2, activation='softmax',
                            input_shape=(input_size, input_size, channels))
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

