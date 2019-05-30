import numpy as np
import copy

class DataPreprocesser(object):
    """
    Will handle image editing.
    """


    def __init__(self, settings, number_of_channels):
        self.settings = settings

        # storing information on how to normalize the data
        self.zeroweighting_L_means_per_channel = []
        self.zeroweighting_L_stds_per_channel = []

        self.zeroweighting_R_means_per_channel = []
        self.zeroweighting_R_stds_per_channel = []

        self.number_of_channels = number_of_channels

    # to do:
    # channel wise normalization
    # - on training dataset
    # - then use the same values on the val dataset

    def process_dataset(self, train, val, test):
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

        number_of_channels = self.number_of_channels

        for channel in range(number_of_channels):
            l = lefts[:, :, :, channel].flatten()
            l_mean = np.mean(l)
            l_std = np.std(l)

            lefts[:, :, :, channel] -= l_mean
            lefts[:, :, :, channel] /= l_std
            val_lefts[:, :, :, channel] -= l_mean
            val_lefts[:, :, :, channel] /= l_std
            test_lefts[:, :, :, channel] -= l_mean
            test_lefts[:, :, :, channel] /= l_std

            self.zeroweighting_L_means_per_channel.append(l_mean)
            self.zeroweighting_L_stds_per_channel.append(l_std)

            r = rights[:, :, :, channel].flatten()
            r_mean = np.mean(r)
            r_std = np.std(r)

            rights[:, :, :, channel] -= r_mean
            rights[:, :, :, channel] /= r_std
            val_rights[:, :, :, channel] -= r_mean
            val_rights[:, :, :, channel] /= r_std
            test_rights[:, :, :, channel] -= r_mean
            test_rights[:, :, :, channel] /= r_std

            self.zeroweighting_R_means_per_channel.append(r_mean)
            self.zeroweighting_R_stds_per_channel.append(r_std)

        train = lefts, rights, labels
        val = val_lefts, val_rights, val_labels
        test = test_lefts, test_rights, test_labels

        #print("Dataset normalization:")
        #print("self.zeroweighting_L_means_per_channel", self.zeroweighting_L_means_per_channel)
        #print("self.zeroweighting_L_stds_per_channel", self.zeroweighting_L_stds_per_channel)
        #print("self.zeroweighting_R_means_per_channel", self.zeroweighting_R_means_per_channel)
        #print("self.zeroweighting_R_stds_per_channel", self.zeroweighting_R_stds_per_channel)

        return [train, val, test]

    def apply_on_a_set_nondestructively(self, set, no_labels = False, be_destructive=False):
        # set can be train, it can be val or anything
        # we don't change the original data, instead we return an edited copy

        if be_destructive:
            set_copy = set
        else:
            set_copy = copy.deepcopy(set)
        if no_labels:
            lefts, rights = set_copy
        else:
            lefts, rights, labels = set_copy

        lefts = np.asarray(lefts).astype('float32')
        rights = np.asarray(rights).astype('float32')

        # insp. https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
        # standartized = (x_np - x_np.mean()) / x_np.std()

        number_of_channels = self.number_of_channels
        for channel in range(number_of_channels):
            l_mean = self.zeroweighting_L_means_per_channel[channel]
            l_std = self.zeroweighting_L_stds_per_channel[channel]

            r_mean = self.zeroweighting_R_means_per_channel[channel]
            r_std = self.zeroweighting_R_stds_per_channel[channel]

            lefts[:, :, :, channel] -= l_mean
            lefts[:, :, :, channel] /= l_std

            rights[:, :, :, channel] -= r_mean
            rights[:, :, :, channel] /= r_std

        if no_labels:
            set_return = lefts, rights
        else:
            set_return = lefts, rights, labels

        return set_return

    def postprocess_images(self, images_L, images_R):
        # from normalized, zero weighted back to the original values

        number_of_channels = self.number_of_channels

        if number_of_channels == 4:
            range_for_just_channels_saved = [1, 2, 3]
        else:
            range_for_just_channels_saved = [0, 1, 2]

        range_for_just_channels_on_images = [0,1,2] # we cut of one channel before

        for channel_i in range(len(range_for_just_channels_saved)):
            channel = range_for_just_channels_saved[channel_i]

            print(self.zeroweighting_L_means_per_channel, channel)
            l_mean = self.zeroweighting_L_means_per_channel[channel]
            l_std = self.zeroweighting_L_stds_per_channel[channel]
            r_mean = self.zeroweighting_R_means_per_channel[channel]
            r_std = self.zeroweighting_R_stds_per_channel[channel]

            # original data underwent x = ((x - xmean)/ xstd)
            # revert by x = (xstd * x) + xmean
            channel_imgs = range_for_just_channels_on_images[channel_i]
            images_L[:, :, :, channel_imgs] = (images_L[:, :, :, channel_imgs] * l_std) + l_mean
            images_R[:, :, :, channel_imgs] = (images_R[:, :, :, channel_imgs] * r_std) + r_mean

        images_L = np.asarray(images_L).astype('uint8')
        images_R = np.asarray(images_R).astype('uint8')

        return images_L, images_R

    # Ye Olde(r) ways
    def process_dataset_OLDSIMPLE(self, dataset):
        lefts, rights, labels = dataset
        # from 0-255 : into -0.5 - 0.5
        #lefts = (lefts / 255.0) - 0.5
        #rights = (rights / 255.0) - 0.5

        # from 0-255 : into 0.0 - 1.0
        lefts = (lefts / 255.0)
        rights = (rights / 255.0)

        # keep at 0-1 for the sigmoid
        #labels = labels - 0.5
        #labels = labels / 2.0

        return [lefts, rights, labels]

    def postprocess_labels(self, labels):
        # serves to project final labels back to where they originally were
        # no need right now, we didn't touch the labels
        return labels

