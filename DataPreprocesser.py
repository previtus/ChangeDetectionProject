from PIL import Image, ImageDraw
import numpy as np

class DataPreprocesser(object):
    """
    Will handle image editing.
    """


    def __init__(self, settings):
        self.settings = settings

    # to do:
    # channel wise normalization
    # - on training dataset
    # - then use the same values on the val dataset

    def process_dataset_Train(self, train, val, test):
        lefts, rights, labels = train
        val_lefts, val_rights, val_labels = val
        test_lefts, test_rights, test_labels = test

        lefts = np.asarray(lefts).astype('float32')
        val_lefts = np.asarray(val_lefts).astype('float32')
        test_lefts = np.asarray(test_lefts).astype('float32')
        rights = np.asarray(rights).astype('float32')
        val_rights = np.asarray(val_rights).astype('float32')
        test_rights = np.asarray(test_rights).astype('float32')

        # insp. https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
        # standartized = (x_np - x_np.mean()) / x_np.std()

        for channel in range(4):
            l = lefts[:, :, :, channel].flatten()
            l_mean = np.mean(l)
            l_std = np.std(l)

            lefts[:, :, :, channel] -= l_mean
            lefts[:, :, :, channel] /= l_std
            val_lefts[:, :, :, channel] -= l_mean
            val_lefts[:, :, :, channel] /= l_std
            test_lefts[:, :, :, channel] -= l_mean
            test_lefts[:, :, :, channel] /= l_std

            r = rights[:, :, :, channel].flatten()
            r_mean = np.mean(r)
            r_std = np.std(r)

            rights[:, :, :, channel] -= r_mean
            rights[:, :, :, channel] /= r_std
            val_rights[:, :, :, channel] -= r_mean
            val_rights[:, :, :, channel] /= r_std
            test_rights[:, :, :, channel] -= r_mean
            test_rights[:, :, :, channel] /= r_std

        train = lefts, rights, labels
        val = val_lefts, val_rights, val_labels
        test = test_lefts, test_rights, test_labels

        return [train, val, test]


    # Ye Olde(r) ways
    def process_dataset(self, dataset):
        lefts, rights, labels = dataset
        # from 0-255 : into 0.0 - 1.0
        lefts = (lefts / 255.0) - 0.5
        rights = (rights / 255.0) - 0.5

        # keep at 0-1 for the sigmoid
        #labels = labels - 0.5
        #labels = labels / 2.0

        return [lefts, rights, labels]

    def postprocess_labels(self, labels):
        # serves to project final labels back to where they originally were

        #labels = (labels + 0.5)
        #labels = labels * 2.0

        return labels

    def postprocess_images(self, images):
        # from -0.5-0.5 back to 0.0-1.0

        images = (images + 0.5)
        # matlibplot at needs positive values, 0.0 -- 1.0

        # hax for now
        #images += 100.0
        #images = np.asarray(images).astype('uint8')

        return images