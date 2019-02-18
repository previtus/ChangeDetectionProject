# FORCE CPU
#import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import Debugger
import keras
from keras.layers import Input, Dense
from keras.models import Model
#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

class Model1_SkipSiamFCN(object):
    """
    An intermediate between the code and bunch of tester models.
    """

    def __init__(self, settings, dataset):
        self.settings = settings
        self.dataset = dataset
        self.debugger = Debugger.Debugger()

        self.model = self.create_model(256)
        self.model.summary()


        self.local_setting_batch_size = 32
        self.local_setting_epochs = 5

    def train(self):
        print("Train")

        lefts, rights, labels = self.dataset

        # 3 channels only - rgb
        lefts = lefts[:,:,:,1:4]
        rights = rights[:,:,:,1:4]
        # label also reshape
        labels = labels.reshape(labels.shape + (1,))

        """
        print("left images")
        self.debugger.explore_set_stats(lefts)
        print("right images")
        self.debugger.explore_set_stats(rights)
        print("label images")
        self.debugger.explore_set_stats(labels)
        """

        split_idx = 4000
        split_test_idx = 4500
        train_X_left = lefts[0:split_idx]
        train_X_right = rights[0:split_idx]
        train_Y = labels[0:split_idx]

        val_X_left = lefts[split_idx:split_test_idx]
        val_X_right = rights[split_idx:split_test_idx]
        val_Y = labels[split_idx:split_test_idx]

        history = self.model.fit([train_X_left, train_X_right], train_Y, batch_size=self.local_setting_batch_size, epochs=self.local_setting_epochs,
                                 validation_data=([val_X_left, val_X_right], val_Y))

        print(history)
        history.history["accuracy"] = history.history["acc"]

        self.debugger.nice_plot_history(history, no_val=True)

    def save(self):
        self.model.save_weights(self.settings.large_file_folder+"last_trained_model_weights.h5")
        print("Saved model weights.")

    def load(self):
        self.model.load_weights(self.settings.large_file_folder+"last_trained_model_weights.h5")
        print("Loaded model weights.")

    def test(self):
        print("Test")

        lefts, rights, labels = self.dataset

        # 3 channels only - rgb
        lefts = lefts[:,:,:,1:4]
        rights = rights[:,:,:,1:4]
        # label also reshape
        labels = labels.reshape(labels.shape + (1,))

        print(lefts.shape)

        split_test_idx = 4500
        split_test_idx = 800
        test_X_left = lefts[split_test_idx:]
        test_X_right = rights[split_test_idx:]
        test_Y = labels[split_test_idx:]

        predicted = self.model.predict([test_X_left, test_X_right])

        # chop off that last dimension
        predicted = predicted.reshape(predicted.shape[:-1])
        test_Y = test_Y.reshape(test_Y.shape[:-1])


        print("left images")
        self.debugger.explore_set_stats(test_X_left)
        print("right images")
        self.debugger.explore_set_stats(test_X_right)
        print("label images")
        self.debugger.explore_set_stats(test_Y)
        print("predicted images")
        self.debugger.explore_set_stats(predicted)

        off = 0
        while off < len(predicted):
            self.debugger.viewTripples(test_X_left, test_X_right, test_Y, how_many=4, off=off)
            self.debugger.viewQuadrupples(test_X_left, test_X_right, test_Y, predicted, how_many=4, off=off)
            off += 4

    def create_model(self, input_size = 112, kernel_size = (3, 3), pool_size = (2, 2), up_size = (2, 2)):
        # input_size = None
        # building blocks:

        # conv + pool
        # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        # keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)

        # deconv + upsample
        # keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        # keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')


        input = Input(shape=(input_size, input_size, 3))
        # 1st block
        x = keras.layers.Conv2D(16, kernel_size, padding="same")(input)
        skip16 = keras.layers.Conv2D(16, kernel_size, padding="same")(x)
        x = keras.layers.MaxPooling2D(pool_size)(skip16)

        # 2nd block
        x = keras.layers.Conv2D(32, kernel_size, padding="same")(x)
        skip32 = keras.layers.Conv2D(32, kernel_size, padding="same")(x)
        x = keras.layers.MaxPooling2D(pool_size)(skip32)

        # 3rd block
        x = keras.layers.Conv2D(64, kernel_size, padding="same")(x)
        x = keras.layers.Conv2D(64, kernel_size, padding="same")(x)
        skip64 = keras.layers.Conv2D(64, kernel_size, padding="same")(x)
        x = keras.layers.MaxPooling2D(pool_size)(skip64)

        # 4th block
        x = keras.layers.Conv2D(128, kernel_size, padding="same")(x)
        x = keras.layers.Conv2D(128, kernel_size, padding="same")(x)
        skip128 = keras.layers.Conv2D(128, kernel_size, padding="same")(x)
        out = keras.layers.MaxPooling2D(pool_size)(skip128)

        # siamese_model = Model(input, out)
        siamese_model_encode = Model(inputs=[input], outputs=[out, skip16, skip32, skip64, skip128])

        print("<< Siamese model >>")
        siamese_model_encode.summary()

        # Then merging
        input_a = Input(shape=(input_size, input_size, 3))
        input_b = Input(shape=(input_size, input_size, 3))

        branch_a, skip16_a, skip32_a, skip64_a, skip128_a = siamese_model_encode([input_a])
        branch_b, skip16_b, skip32_b, skip64_b, skip128_b = siamese_model_encode([input_b])

        # merge two branches
        x = keras.layers.concatenate([branch_a, branch_a])
        x = keras.layers.Conv2DTranspose(128, kernel_size, padding="same")(x)
        x = keras.layers.UpSampling2D(up_size)(x)

        # merge 128
        x = keras.layers.concatenate([x, skip128_a, skip128_b])

        x = keras.layers.Conv2DTranspose(128, kernel_size, padding="same")(x)
        x = keras.layers.Conv2DTranspose(128, kernel_size, padding="same")(x)
        x = keras.layers.Conv2DTranspose(64, kernel_size, padding="same")(x)
        x = keras.layers.UpSampling2D(up_size)(x)

        # merge 64
        x = keras.layers.concatenate([x, skip64_a, skip64_b])
        x = keras.layers.Conv2DTranspose(64, kernel_size, padding="same")(x)
        x = keras.layers.Conv2DTranspose(64, kernel_size, padding="same")(x)
        x = keras.layers.Conv2DTranspose(32, kernel_size, padding="same")(x)
        x = keras.layers.UpSampling2D(up_size)(x)

        # merge 32
        x = keras.layers.concatenate([x, skip32_a, skip32_b])
        x = keras.layers.Conv2DTranspose(32, kernel_size, padding="same")(x)
        x = keras.layers.Conv2DTranspose(16, kernel_size, padding="same")(x)
        x = keras.layers.UpSampling2D(up_size)(x)

        # merge 16
        x = keras.layers.concatenate([x, skip16_a, skip16_b])
        x = keras.layers.Conv2DTranspose(16, kernel_size, padding="same")(x)
        final_out = keras.layers.Conv2DTranspose(2, kernel_size, padding="same")(x)

        # within 0-1
        # and just 1 channel
        final_out = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(final_out)

        print("<< Full model >>")
        full_model = Model([input_a, input_b], final_out)

        #full_model.summary()

        model = full_model
        # model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])
        # model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

        return model



