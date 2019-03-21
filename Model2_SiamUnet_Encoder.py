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

        BACKBONE = 'resnet34'

        resolution_of_input = self.dataset.datasetInstance.IMAGE_RESOLUTION
        self.model = self.create_model(backbone=BACKBONE, input_size = resolution_of_input, channels = 3)
        self.model.summary()

        self.local_setting_batch_size = 32 #32
        self.local_setting_epochs = 30 #100

        self.train_data_augmentation = False # todo possibly ...


    def train(self):
        print("Train")

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


    def test_show_on_train_data_to_see_overfit(self, evaluator):
        print("Debug, showing performance on Train data!")


    def create_model(self, backbone='resnet34', input_size = 112, channels = 3):

        #custom_weights_file = "/scratch/ruzicka/python_projects_large/AerialNet_VariousTasks/model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep_dsm01proper.h5"
        custom_weights_file = "imagenet"

        model = SiameseUnet(backbone, encoder_weights=custom_weights_file, classes=3, activation='softmax',
                            input_shape=(input_size, input_size, channels))
        print("Model loaded:")
        print("model.input", model.input)
        print("model.output", model.output)


        return model

