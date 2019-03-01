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

class Model1_SkipSiamFCN(object):
    """
    An intermediate between the code and bunch of tester models.
    """

    def __init__(self, settings, dataset):
        self.settings = settings
        self.dataset = dataset
        self.dataPreprocesser = dataset.dataPreprocesser
        self.debugger = Debugger.Debugger(settings)

        self.use_sigmoid_or_softmax = 'sigmoid' # softmax is for multiple categories

        self.model = self.create_model(input_size = None, channels = 3)
        self.model.summary()

        self.local_setting_batch_size = 32 #32
        self.local_setting_epochs = 30 #100

        self.train_data_augmentation = False


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

        # Manual inspections, some labels are "weird"
        """
        # 1639 1909
        inspect_i = 1639
        left_path = self.dataset.train_paths[0][inspect_i].split("/")[-1]
        right_path = self.dataset.train_paths[1][inspect_i].split("/")[-1]
        vec_path = self.dataset.train_paths[2][inspect_i].split("/")[-1]
        txts = ["\n\n\n" + left_path + " | " + right_path  + " | " + vec_path]
        self.debugger.viewTripples([train_L[inspect_i]], [train_R[inspect_i]], [train_V[inspect_i, :, :, 0]],txts=txts,how_many=1, off=0)
        # doublecheck
        self.debugger.viewTrippleFromUrl(self.dataset.train_paths[0][inspect_i], self.dataset.train_paths[1][inspect_i], self.dataset.train_paths[2][inspect_i])
        """

        if self.train_data_augmentation:
            # Training with data augmentation:
            from keras.preprocessing.image import ImageDataGenerator
            data_gen_args = dict(#featurewise_center=True,
                                 #featurewise_std_normalization=True,
                                 #rotation_range=90,
                                 #width_shift_range=0.1,
                                 #height_shift_range=0.1,
                                 #zoom_range=0.2,

                                 horizontal_flip=True,
                                 vertical_flip=True
                                 )
            datagen_L = ImageDataGenerator(**data_gen_args)
            datagen_R = ImageDataGenerator(**data_gen_args)
            datagen_V = ImageDataGenerator(**data_gen_args)

            sync_labels = range(len(train_L))

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            #datagen_L.fit(train_L)
            seed = 1
            datagen_L.fit(train_L, augment=True, seed=seed)
            datagen_R.fit(train_R, augment=True, seed=seed)
            datagen_V.fit(train_V, augment=True, seed=seed)

            class History(object):
                history = {}

            history = History()
            history.history = {}
            history.history["loss"] = []
            history.history["acc"] = []
            history.history["val_loss"] = []
            history.history["val_acc"] = []


            for e in range(self.local_setting_epochs):
                print('Epoch', e)
                batches = 0

                datagen_L_generator = datagen_L.flow(train_L, sync_labels, batch_size=self.local_setting_batch_size, seed=seed)
                datagen_R_generator = datagen_R.flow(train_R, sync_labels, batch_size=self.local_setting_batch_size, seed=seed)
                datagen_V_generator = datagen_V.flow(train_V, sync_labels, batch_size=self.local_setting_batch_size, seed=seed)

                for l_batch, l_idx_batch in datagen_L_generator:
                    r_batch, r_idx_batch = next(datagen_R_generator)
                    v_batch, v_idx_batch = next(datagen_V_generator)

                    if self.settings.verbose >= 3:
                        print("--- another batch ---")
                        print(l_idx_batch, l_batch.shape)
                        print(r_idx_batch, r_batch.shape)
                        print(v_idx_batch, v_batch.shape)

                        self.debugger.viewTripples(l_batch, r_batch, v_batch[:,:,:,0], how_many=4, off=0)
                        self.debugger.viewTripples(train_L[l_idx_batch], train_R[r_idx_batch], train_V[v_idx_batch][:,:,:,0], how_many=4, off=0)

                    h = self.model.fit([l_batch, r_batch], v_batch, verbose=0)

                    #self.model.fit(x_batch, y_batch)
                    batches += 1
                    if batches >= len(train_L) / 32:
                        # we need to break the loop by hand because
                        # the generator loops indefinitely
                        break

                # add validation measurement to histories?
                metrics = self.model.evaluate(x=[val_L, val_R], y=val_V, verbose=0)
                print("val",list(zip(self.model.metrics_names, metrics)))
                history.history["val_loss"].append(metrics[0]) # 'loss', 'binary_accuracy', 'mean_squared_error'
                history.history["val_acc"].append(metrics[1])

                # we can evaluate on the original training data - right now it's without the augments (hmmm ... maybe I should keep them and then eval on them instead?)
                metrics_train = self.model.evaluate(x=[train_L, train_R], y=train_V, verbose=0)
                print("train",list(zip(self.model.metrics_names, metrics_train)))
                history.history["loss"].append(metrics_train[0])
                history.history["acc"].append(metrics_train[1])

        else:
            # Regular training:
            history = self.model.fit([train_L, train_R], train_V, batch_size=self.local_setting_batch_size, epochs=self.local_setting_epochs,
                                     validation_data=([val_L, val_R], val_V))
            print(history.history)
            if self.use_sigmoid_or_softmax == 'sigmoid':
                history.history["acc"] = history.history["binary_accuracy"]
                history.history["val_acc"] = history.history["val_binary_accuracy"]
            else:
                history.history["acc"] = history.history["categorical_accuracy"]
                history.history["val_acc"] = history.history["val_categorical_accuracy"]
            print(history.history)


        self.debugger.nice_plot_history(history)


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

        predicted = self.model.predict([test_L, test_R])
        metrics = self.model.evaluate(x=[test_L, test_R], y=test_V_cat, verbose=0)
        metrics_info = self.model.metrics_names
        print(list(zip(metrics_info, metrics)))

        if self.use_sigmoid_or_softmax == 'softmax':
            #print("Predicted is now in this shape:")
            #print("predicted.shape:", predicted.shape)
            #self.debugger.viewVectors(predicted[:,:,:,1])
            #self.debugger.viewVectors(predicted[:,:,:,0])

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

        # Evaluator
        #evaluator.histogram_of_predictions(predicted)
        print("threshold=0.5")
        evaluator.calculate_metrics(predicted, test_V)
        print("threshold=0.1")
        evaluator.calculate_metrics(predicted, test_V, threshold=0.1)
        print("threshold=0.05")
        evaluator.calculate_metrics(predicted, test_V, threshold=0.05)
        print("threshold=0.01")
        evaluator.calculate_metrics(predicted, test_V, threshold=0.01)

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

        test_L, test_R, test_V = self.dataset.train

        test_L = test_L[0:200]
        test_R = test_R[0:200]
        test_V = test_V[0:200]

        if test_L.shape[3] > 3:
            test_L = test_L[:,:,:,1:4]
            test_R = test_R[:,:,:,1:4]
        if self.use_sigmoid_or_softmax == 'softmax':
            test_V_cat = to_categorical(test_V)
        else:
            test_V_cat = test_V.reshape(test_V.shape + (1,))

        predicted = self.model.predict([test_L, test_R])
        metrics = self.model.evaluate(x=[test_L, test_R], y=test_V_cat, verbose=0)
        metrics_info = self.model.metrics_names
        print(list(zip(metrics_info, metrics)))

        if self.use_sigmoid_or_softmax == 'softmax':
            predicted = predicted[:, :, :, 1]
        else:
            predicted = predicted.reshape(predicted.shape[:-1])

        predicted = self.dataPreprocesser.postprocess_labels(predicted)
        # Evaluator
        #evaluator.histogram_of_predictions(predicted)
        evaluator.calculate_metrics(predicted, test_V)

        test_L, test_R = self.dataPreprocesser.postprocess_images(test_L, test_R)

        if test_L.shape[3] > 3:
            # 3 channels only - rgb
            test_L = test_L[:,:,:,1:4]
            test_R = test_R[:,:,:,1:4]

        off = 0
        while off < len(predicted):
            #self.debugger.viewTripples(test_L, test_R, test_V, how_many=4, off=off)
            self.debugger.viewQuadrupples(test_L, test_R, test_V, predicted, how_many=4, off=off)
            off += 4

    def create_model(self, input_size = 112, channels = 3, kernel_size = (3, 3), pool_size = (2, 2), up_size = (2, 2)):
        # input_size = None
        # building blocks:

        # conv + pool
        # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        # keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)

        # deconv + upsample
        # keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        # keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')


        input = Input(shape=(input_size, input_size, channels))
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
        input_a = Input(shape=(input_size, input_size, channels))
        input_b = Input(shape=(input_size, input_size, channels))

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
        if self.use_sigmoid_or_softmax == 'softmax':
            activ = "softmax"
            loss = "categorical_crossentropy"
            metric = "categorical_accuracy"

            classes_out = 2 #now at least, only 0 or 1
        else:
            activ = "sigmoid"
            loss = "binary_crossentropy"
            metric = "binary_accuracy"
            classes_out = 1  # between 0-1 its a real num

        final_out = keras.layers.Conv2D(classes_out, (1, 1), activation=activ)(final_out)

        print("<< Full model >>")
        full_model = Model([input_a, input_b], final_out)

        #full_model.summary()

        model = full_model
        # model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])
        # model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        """ Experiment
        # with the contrastive loss the labeling was done in other way around!, which gives us the opposite predictions
        from keras import backend as K
        def contrastive_loss(y_true, y_pred):
            '''Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            '''
            margin = 1
            sqaure_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - y_pred, 0))
            return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
        """

        model.compile(optimizer=Adam(lr=0.00001), loss=loss, metrics=[metric, 'mse'])

        # output dim depending on the type of a problem: https://stats.stackexchange.com/questions/246287/regarding-the-output-format-for-semantic-segmentation
        # Either: last activation sigmoid +  binary_crossentropy loss
        # Or: have 2 classes (0s and 1s) converted by "train_V = to_categorical(train_V)" and output of size 2 and as a softmax


        return model



