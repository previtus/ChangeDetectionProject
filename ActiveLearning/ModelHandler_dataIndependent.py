# FORCE CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import Debugger
from keras.optimizers import Adam
from loss_weighted_crossentropy import weighted_categorical_crossentropy
from Model2_builder import SiameseUnet

class ModelHandler_dataIndependent(object):
    """
    An intermediate between the code and bunch of tester models.
    """

    def __init__(self, settings, BACKBONE = 'resnet34', verbose=1):
        self.settings = settings
        self.debugger = Debugger.Debugger(settings)
        self.verbose = verbose

        self.use_sigmoid_or_softmax = 'softmax'
        assert self.use_sigmoid_or_softmax == 'softmax'

        #BACKBONE = 'resnet34'
        #BACKBONE = 'resnet50' #batch 16
        #BACKBONE = 'resnet101' #batch 8
        #BACKBONE = 'seresnext50' #trying batch 16 as well
        custom_weights_file = "imagenet"

        #weights from imagenet finetuned on aerial data specific task - will it work? will it break?
        #custom_weights_file = "/scratch/ruzicka/python_projects_large/AerialNet_VariousTasks/model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep_dsm01proper.h5"

        resolution_of_input = None
        #resolution_of_input = 256
        self.model = self.create_model(backbone=BACKBONE, custom_weights_file=custom_weights_file, input_size = resolution_of_input, channels = 3)

        if self.verbose >= 2:
            self.model.summary()

    def create_model(self, backbone='resnet34', custom_weights_file = "imagenet", input_size = 112, channels = 3):

        model = SiameseUnet(backbone, encoder_weights=custom_weights_file, classes=2, activation='softmax',
                            input_shape=(input_size, input_size, channels), encoder_freeze=False)

        if self.verbose >= 2:
            print("Model loaded:")
            print("model.input", model.input)
            print("model.output", model.output)

        # Loss and metrics:

        #loss = "categorical_crossentropy"

        weights = [1, 3]
        loss = weighted_categorical_crossentropy(weights)

        metric = "categorical_accuracy"
        model.compile(optimizer=Adam(lr=0.00001), loss=loss, metrics=[metric, 'mse'])

        return model


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

