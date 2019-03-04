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

class Model1b_SkipSiamFCN_withClassLabel(object):
    """
    An intermediate between the code and bunch of tester models.
    """

    def __init__(self, settings, dataset):
        self.settings = settings
        self.dataset = dataset
        self.dataPreprocesser = dataset.dataPreprocesser
        self.debugger = Debugger.Debugger(settings)

        self.use_sigmoid_or_softmax = 'sigmoid' # softmax is for multiple categories

        resolution_of_input = self.dataset.datasetInstance.variant

        self.model = self.create_model(input_size = resolution_of_input, channels = 3)
        self.model.summary()

        self.local_setting_batch_size = 32 #32
        self.local_setting_epochs = 10 #100

        self.train_data_augmentation = False


    def train(self):
        print("Train")

        train_L, train_R, train_V = self.dataset.train
        val_L, val_R, val_V = self.dataset.val

        train_class_Y = self.dataset.train_classlabels
        val_class_Y = self.dataset.val_classlabels

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

        added_plots = []

        if self.train_data_augmentation:
            print("FOOO")

        else:
            # change weights of each class?
            # not sure if this one will work with 2D img as label
            # class_weight = {0: 1.0, 1: 1.0} #
            # so maybe
            # sample_weight: optional array of the same length as x, containing weights to apply to the model's loss for each sample
            #       give it a rebalancing?


            # Regular training:
            history = self.model.fit([train_L, train_R], [train_V, train_class_Y], batch_size=self.local_setting_batch_size, epochs=self.local_setting_epochs,
                                     validation_data=([val_L, val_R], [val_V, val_class_Y]), verbose=2) # 2 ~ 1 line each ep
            print(history.history)
            if self.use_sigmoid_or_softmax == 'sigmoid':
                # {'val_loss': [0.5298519569635392],
                # 'val_conv2d_11_loss': [0.4634183657169342],
                # 'val_dense_4_loss': [0.5962855698540807],
                # 'val_conv2d_11_binary_accuracy': [0.83706791639328],
                # 'val_conv2d_11_mean_squared_error': [0.14408911854028703],
                # 'val_dense_4_binary_accuracy': [0.3],
                # 'val_dense_4_mean_squared_error': [2.1713935780525206],

                # 'loss': [1.8971326723965731],
                # 'conv2d_11_loss': [0.5563809064301577],
                # 'dense_4_loss': [3.2378844252499666],
                #
                # 'conv2d_11_binary_accuracy': [0.7394221806526184],
                # 'conv2d_11_mean_squared_error': [0.18496516672047703],
                # 'dense_4_binary_accuracy': [0.2772727272727273],
                # 'dense_4_mean_squared_error': [2.65305721282959]}
                # label and mask_acc
                history.history["acc"] = history.history["label_binary_accuracy"]  # we care about this one to show
                history.history["val_acc"] = history.history["val_label_binary_accuracy"]
                history.history["mask_acc"] = history.history["mask_binary_accuracy"]
                history.history["val_mask_acc"] = history.history["val_mask_binary_accuracy"]
                added_plots = ["mask_acc","val_mask_acc"]
            else:
                history.history["acc"] = history.history["label_categorical_accuracy"]
                history.history["val_acc"] = history.history["val_label_categorical_accuracy"]
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
        test_class_Y = self.dataset.test_classlabels

        if test_L.shape[3] > 3:
            # 3 channels only - rgb
            test_L = test_L[:,:,:,1:4]
            test_R = test_R[:,:,:,1:4]
        # label also reshape
        if self.use_sigmoid_or_softmax == 'softmax':
            test_V_cat = to_categorical(test_V)
        else:
            test_V_cat = test_V.reshape(test_V.shape + (1,))

        predicted_masks, predicted_labels = self.model.predict([test_L, test_R])
        predicted = predicted_masks
        metrics = self.model.evaluate(x=[test_L, test_R], y=[test_V_cat, test_class_Y], verbose=0)
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
            predicted_labels = predicted_labels.reshape(predicted_labels.shape[:-1])

        # undo preprocessing steps?
        predicted = self.dataPreprocesser.postprocess_labels(predicted)

        print("CLASS EVALUATION")
        print("this:")
        print(len(predicted_labels))
        print(predicted_labels.shape)
        print("with that:")
        print(len(test_class_Y))
        print(test_class_Y.shape)


        evaluator.histogram_of_predictions(predicted_labels)
        evaluator.histogram_of_predictions(test_class_Y)

        print("threshold=0.5")
        predictions_thresholded = evaluator.calculate_metrics(predicted_labels, test_class_Y)
        predictions_thresholded = predictions_thresholded[0].astype(int)
        print("threshold=0.1")
        predictions_thresholded = evaluator.calculate_metrics(predicted_labels, test_class_Y, threshold=0.1)
        predictions_thresholded = predictions_thresholded[0].astype(int)

        txts = []
        for i in range(len(test_class_Y)):
            pred = predictions_thresholded[i]
            gt = test_class_Y[i]
            txt = "gt "+str(gt)+" pred "+str(pred)+"\n"
            txts.append(txt)

        print("MASK EVALUATION")
        # Evaluator
        #evaluator.histogram_of_predictions(predicted)
        print("threshold=0.5")
        evaluator.calculate_metrics(predicted, test_V)
        print("threshold=0.1")
        evaluator.calculate_metrics(predicted, test_V, threshold=0.1)
        print("threshold=0.05")
        evaluator.calculate_metrics(predicted, test_V, threshold=0.05)
        #print("threshold=0.01")
        #evaluator.calculate_metrics(predicted, test_V, threshold=0.01)

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
            self.debugger.viewQuadrupples(test_L, test_R, test_V, predicted, txts, how_many=4, off=off)
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

        # out comes:
        #         112 #max_pooling2d_4 (MaxPooling2 (None, 7, 7, 128)
        #         256 #max_pooling2d_4(MaxPooling2(None, 16, 16, 128)
        #

        print("<< Siamese model >>")
        siamese_model_encode.summary()

        # Then merging
        input_a = Input(shape=(input_size, input_size, channels))
        input_b = Input(shape=(input_size, input_size, channels))

        branch_a, skip16_a, skip32_a, skip64_a, skip128_a = siamese_model_encode([input_a])
        branch_b, skip16_b, skip32_b, skip64_b, skip128_b = siamese_model_encode([input_b])

        # merge two branches
        joined_siam_x = keras.layers.concatenate([branch_a, branch_b])
        x = keras.layers.Conv2DTranspose(128, kernel_size, padding="same")(joined_siam_x)
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

        final_out = keras.layers.Conv2D(classes_out, (1, 1), activation=activ, name="mask")(final_out)

        classifier = keras.layers.Flatten()(joined_siam_x)
        classifier = keras.layers.Dense(2048)(classifier)
        classifier = keras.layers.Dropout(0.5)(classifier)
        classifier = keras.layers.Dense(256)(classifier)
        classifier = keras.layers.Dropout(0.5)(classifier)
        classifier = keras.layers.Dense(32)(classifier)
        classifier_out = keras.layers.Dense(1, activation="sigmoid", name="label")(classifier)

        print("<< Full model >>")
        full_model = Model(inputs = [input_a, input_b], outputs=[final_out, classifier_out])

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

        model.compile(optimizer=Adam(lr=0.00001), loss=loss, metrics=[metric, 'mse'], loss_weights=[0.5, 0.5])

        # output dim depending on the type of a problem: https://stats.stackexchange.com/questions/246287/regarding-the-output-format-for-semantic-segmentation
        # Either: last activation sigmoid +  binary_crossentropy loss
        # Or: have 2 classes (0s and 1s) converted by "train_V = to_categorical(train_V)" and output of size 2 and as a softmax


        return model



