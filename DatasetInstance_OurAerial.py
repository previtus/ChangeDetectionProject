import Debugger

import os
import random
from os import listdir
from os.path import isfile, join
import re
import pickle
import numpy as np
from skimage import io
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt


class DatasetInstance_OurAerial(object):
    """
    Contains specific setting for one dataset instance.

    Has/Can do these:
    - Has paths, file names
    - data specific edits (when one dataset goes from [1 to 2] with it labels without explanations)
    - can have several variants (setting options)
    - specific paths to saved files
    - print to present itself
    """


    def __init__(self, settings, dataLoader, variant = "256"):
        self.settings = settings
        self.dataLoader = dataLoader
        self.debugger = Debugger.Debugger(settings)
        self.DEBUG_TURN_OFF_BALANCING = False

        self.variant = variant # 256 = 256x256, 112 = 112x112

        self.local_setting_skip_rows = 2
        self.local_setting_skip_columns = 2

        self.save_path_ = "OurAerial_preloadedImgs_sub"

        if self.variant == "256" or self.variant == "256_clean" or self.variant == "256_cleanManual_noOver" or self.variant == "256_cleanManual":
            self.dataset_version = "256x256_over32"
            if self.variant == "256_clean":
                self.dataset_version = "256x256_over32_clean"
            if self.variant == "256_cleanManual_noOver": # <<<< currently no data
                self.dataset_version = "256x256_cleanManual_noOver"
            if self.variant == "256_cleanManual":
                self.dataset_version = "256x256_cleanManual"

                # possibly can have a more generous self.bigger_than_percent ... ?

            #self.SUBSET = 83000
            self.SUBSET = -1
            #self.SUBSET = 1000
            self.IMAGE_RESOLUTION = 256
            self.CHANNEL_NUMBER = 4
            self.LOAD_BATCH_INCREMENT = 10000 # loads in this big batches for each balancing

            self.bigger_than_percent = 8.0  # 8.0 from full set
            self.smaller_than_percent = 1.0  # 3.0 ?

            self.default_raster_shape = (256,256,4)
            self.default_vector_shape = (256,256)

            # decent dataset:
            self.hdf5_path = self.settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL8.0_1.0_sel1428_res256x256.h5"

            # spliting <1428>
            # 1200 train, 100 val, 128 test
            self.split_train = 1200
            self.split_val = 1300

            if self.variant == "256_clean":
                self.hdf5_path = self.settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL8.0_1.0_sel1086_res256x256.h5"
                # 1086 in total
                self.split_train = 900
                self.split_val = 1000

            if self.variant == "256_cleanManual":
                # needs also the source images without overlap!
                #self.hdf5_path = self.settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL3.0_1.0_sel2144_res256x256.h5"
                self.hdf5_path = self.settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL3.0_1.0_sel2144_res256x256_SMALLER.h5"
                # 8 perc => 942/2 change images
                # 3 perc => 2144/2 change images
                self.bigger_than_percent = 3.0 # try?
                self.smaller_than_percent = 1.0  # there shouldn't be much noise in this ...

                self.split_train = 1900
                self.split_val = 2000


        elif self.variant == "112" or self.variant == "112_clean":
            self.dataset_version = "112x112"
            if self.variant == "112_clean":
                self.dataset_version = "112x112_clean"
            #self.SUBSET = 118667
            self.SUBSET = -1
            self.LOAD_BATCH_INCREMENT = 100000


            self.IMAGE_RESOLUTION = 112
            self.CHANNEL_NUMBER = 4

            self.bigger_than_percent = 18.0  # 18.0
            self.smaller_than_percent = 1.0  # 5.0

            self.default_raster_shape = (112, 112, 4)
            self.default_vector_shape = (112, 112)

            # decent dataset:
            self.hdf5_path = self.settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL18.0_1.0_sel2380_res112x112.h5"

            # spliting <2380>
            # 2200 train, 100 val, 80 test
            self.split_train = 2200
            self.split_val = 2300

            if self.variant == "112_clean":
                self.hdf5_path = self.settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL18.0_1.0_sel2212_res112x112.h5"
                # 2212 in total
                self.split_train = 2000
                self.split_val = 2100

        elif self.variant == "6368_special":
            self.local_setting_skip_rows = 0
            self.local_setting_skip_columns = 0

            self.dataset_version = "6368_special"
            self.SUBSET = None #all
            self.IMAGE_RESOLUTION = 6368
            self.CHANNEL_NUMBER = 4
            self.LOAD_BATCH_INCREMENT = 20 # from 14 images

            self.bigger_than_percent = 0.0  # doesn't make much sense here!
            self.smaller_than_percent = 0.0  # doesn't make much sense here!

            self.default_raster_shape = (6368,6368,4)
            self.default_vector_shape = (6368,6368)

            # decent dataset:
            self.hdf5_path = self.settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL0.0_0.0_sel13_res6368x6368.h5"

            # spliting <14>
            # 0 train, 0 val, 14 test
            self.split_train = 0
            self.split_val = 0
            self.DEBUG_TURN_OFF_BALANCING = True



    def split_train_val_test_KFOLDCROSSVAL(self, data, test_fold = 0, K = 4):
        lefts, rights, labels = data

        # now we would like the val jump around the dataset (and the rest can be still separated into train - val

        # split [0 - end] into K folds, one as a test the rest as a train (alt. val, but that can be 0)
        N = len(lefts)
        jump_by = int(N / K)

        test_L = np.empty(((0,)+lefts.shape[1:]), lefts.dtype)
        train_L = np.empty(((0,)+lefts.shape[1:]), lefts.dtype)
        val_L = np.empty(((0,)+lefts.shape[1:]), lefts.dtype)
        test_R = np.empty(((0,) + rights.shape[1:]), rights.dtype)
        train_R = np.empty(((0,) + rights.shape[1:]), rights.dtype)
        val_R = np.empty(((0,) + rights.shape[1:]), rights.dtype)
        test_V = np.empty(((0,) + labels.shape[1:]), labels.dtype)
        train_V = np.empty(((0,) + labels.shape[1:]), labels.dtype)
        val_V = np.empty(((0,) + labels.shape[1:]), labels.dtype)

        data_start = 0
        for fold_index in range(K):
            data_until = data_start + jump_by
            if data_until > N:
                data_until = N

            fold_L = lefts[data_start:data_until]
            fold_R = rights[data_start:data_until]
            fold_V = labels[data_start:data_until]

            #print("fold_L.shape", fold_L.shape)

            if fold_index == test_fold:
                # we want to have half test and half val:
                mid = int(len(fold_L)/2)

                test_fold_L = fold_L[mid:]
                val_fold_L = fold_L[0:mid]
                test_fold_R = fold_R[mid:]
                val_fold_R = fold_R[0:mid]
                test_fold_V = fold_V[mid:]
                val_fold_V = fold_V[0:mid]

                # add to test set
                test_L = np.append(test_L, test_fold_L, 0)
                test_R = np.append(test_R, test_fold_R, 0)
                test_V = np.append(test_V, test_fold_V, 0)
                val_L = np.append(val_L, val_fold_L, 0)
                val_R = np.append(val_R, val_fold_R, 0)
                val_V = np.append(val_V, val_fold_V, 0)
            else:
                # add to train set
                train_L = np.append(train_L, fold_L, 0)
                train_R = np.append(train_R, fold_R, 0)
                train_V = np.append(train_V, fold_V, 0)

            data_start += jump_by

        train = [train_L, train_R, train_V]
        test = [test_L, test_R, test_V]
        val = [val_L, val_R, val_V]

        return train, val, test


    def split_train_val_test(self, data):
        lefts, rights, labels = data

        # split [0 : split_train] [split_train : split_val] [split_val : end]

        train_L = lefts[0:self.split_train]
        train_R = rights[0:self.split_train]
        train_V = labels[0:self.split_train]

        val_L = lefts[self.split_train:self.split_val]
        val_R = rights[self.split_train:self.split_val]
        val_V = labels[self.split_train:self.split_val]

        test_L = lefts[self.split_val:]
        test_R = rights[self.split_val:]
        test_V = labels[self.split_val:]

        train = [train_L, train_R, train_V]
        val = [val_L, val_R, val_V]
        test = [test_L, test_R, test_V]

        return train, val, test

    #def datasetSpecificEdit_rasters(self,data):
    #    return data
    #def datasetSpecificEdit_vectors(self,data):
    #    return data

    def present_thyself(self):
        print("Our own dataset of aerial photos. Resolution goes in the variants of 256x256x4 and 112x112x4 (channels: near infra, r,g,b).")


    def load_dataset(self):
        load_paths_from_folders = False  # TRUE To recompute the paths from folder
        load_images_anew = False         # TRUE To reload images from the files directly + rebalance them

        # load_image_paths()
        # save_image_paths_to_cache()

        if load_paths_from_folders:
            # Load paths
            print("\nLoading all paths from input folders:")
            lefts_paths, rights_paths, labels_paths = self.load_paths_from_folders()
            self.dataLoader.save_paths(lefts_paths, self.settings.large_file_folder + "saved_paths_2012_"+self.dataset_version+"ALL.pickle")
            self.dataLoader.save_paths(rights_paths, self.settings.large_file_folder + "saved_paths_2015_"+self.dataset_version+"ALL.pickle")
            self.dataLoader.save_paths(labels_paths, self.settings.large_file_folder + "saved_paths_vectors_"+self.dataset_version+"ALL.pickle")
        else:
            lefts_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "saved_paths_2012_"+self.dataset_version+"ALL.pickle")
            rights_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "saved_paths_2015_"+self.dataset_version+"ALL.pickle")
            labels_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "saved_paths_vectors_"+self.dataset_version+"ALL.pickle")

        print("We have", len(lefts_paths), "2012 images, ", lefts_paths[0:4])
        print("We have", len(rights_paths), "2015 images, ", rights_paths[0:4])
        print("We have", len(labels_paths), "vector images, ", labels_paths[0:4])

        # Load images
        lefts_paths = lefts_paths[0:self.SUBSET]
        rights_paths = rights_paths[0:self.SUBSET]
        labels_paths = labels_paths[0:self.SUBSET]

        # DEBUG SECTION, check if the images of left-right-label correspond to each other!
        """
        for i in range(len(lefts_paths)):
            if labels_paths[i]:
                print(i)

                print(lefts_paths[i])
                print(rights_paths[i])
                print(labels_paths[i])

                self.debugger.viewTrippleFromUrl(lefts_paths[i], rights_paths[i], labels_paths[i])

                print("--------------/n")
        """

        # check_valid_images() + balance_images()
        # save_valid_and_balanced_paths_to_cache()

        if load_images_anew:
            # Load data

            print("\nLoading vector images:")

            total = len(labels_paths)
            batch_i = 0

            overAllBatches_lefts_paths = []
            overAllBatches_rights_paths = []
            overAllBatches_labels_paths = []

            print("\nLoading vector images (in batches):")
            while batch_i < total:
                inc = np.min([total-batch_i, self.LOAD_BATCH_INCREMENT])
                limits = [batch_i, batch_i+inc]
                print("Batch limits:", limits)

                labels_batch = []

                V = []
                for i in range(limits[0], limits[1]):
                    V.append(labels_paths[i])
                L = []
                for i in range(limits[0], limits[1]):
                    L.append(lefts_paths[i])
                R = []
                for i in range(limits[0], limits[1]):
                    R.append(rights_paths[i])

                for path in tqdm( V ):
                    labels_batch.append(self.load_vector_image(path))

                if not self.DEBUG_TURN_OFF_BALANCING:
                    new_lefts_paths, new_rights_paths, new_labels_paths = self.balance_data(labels_batch, L, R, V)
                else:
                    new_lefts_paths, new_rights_paths, new_labels_paths = L, R, V

                #print("Checking paths from the batch:")
                #self.debugger.check_paths(new_lefts_paths, new_rights_paths, new_labels_paths)

                for i in range(len(new_labels_paths)):
                    overAllBatches_labels_paths.append(new_labels_paths[i])
                for i in range(len(new_lefts_paths)):
                    overAllBatches_lefts_paths.append(new_lefts_paths[i])
                for i in range(len(new_rights_paths)):
                    overAllBatches_rights_paths.append(new_rights_paths[i])

                batch_i += inc

            lefts_paths = overAllBatches_lefts_paths
            rights_paths = overAllBatches_rights_paths
            labels_paths = overAllBatches_labels_paths

            #print("Checking paths after batches concatted them:")
            #self.debugger.check_paths(lefts_paths, rights_paths, labels_paths)

            print("\nLoading balanced set of raster images:")
            new_lefts = []
            for path in tqdm(lefts_paths):
                new_lefts.append(self.load_raster_image(path))
            lefts = new_lefts
            new_rights = []
            for path in tqdm(rights_paths):
                new_rights.append(self.load_raster_image(path))
            rights = new_rights
            new_labels = []
            for path in tqdm(labels_paths):
                new_labels.append(self.load_vector_image(path))
            labels = new_labels

            lefts, rights, labels, lefts_paths, rights_paths, labels_paths = self.check_shapes(lefts, rights, labels, lefts_paths, rights_paths, labels_paths)

            #print("Now it should still be the same as the last one ^^^ ")
            #self.check_balance_of_data(labels, labels_paths)

            labels = np.asarray(labels).astype('float32')
            rights = np.asarray(rights).astype('float32')
            lefts = np.asarray(lefts).astype('float32')

            # Save
            self.dataLoader.save_images_to_h5(lefts, rights, labels, self.save_path_+"BAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+"_sel")
            lefts = np.asarray(lefts).astype('uint8')
            rights = np.asarray(rights).astype('uint8')

            name = str(self.SUBSET)+"_"+self.dataset_version+"BALVAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+".pickle"
            self.dataLoader.save_paths(lefts_paths, self.settings.large_file_folder + "LEFT_"+name)
            self.dataLoader.save_paths(rights_paths, self.settings.large_file_folder + "RIGHT_"+name)
            self.dataLoader.save_paths(labels_paths, self.settings.large_file_folder + "LABELS_"+name)

        else:
            # These loaded images are valid (all the same resolution) and balanced according to the setting.

            print("loading images such as:", self.hdf5_path)

            lefts, rights, labels = self.dataLoader.load_images_from_h5(self.hdf5_path)
            lefts = np.asarray(lefts).astype('uint8')
            rights = np.asarray(rights).astype('uint8')
            labels = np.asarray(labels).astype('float32')

            print("loading paths such as:", self.settings.large_file_folder + "saved_paths_2012_"+self.dataset_version+"BALVAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+".pickle")

            name = str(self.SUBSET)+"_"+self.dataset_version+"BALVAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+".pickle"

            lefts_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "LEFT_" + name)
            rights_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "RIGHT_" + name)
            labels_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "LABELS_" + name)

            # test that they are the same
            """
            print("\n.... reloading images .... ")
            new_lefts = []
            new_rights = []
            new_labels = []
            for path in tqdm(labels_paths):
                new_labels.append(self.load_vector_image(path))
            for path in tqdm(lefts_paths):
                new_lefts.append(self.load_raster_image(path))
            for path in tqdm(rights_paths):
                new_rights.append(self.load_raster_image(path))
            new_lefts = np.asarray(new_lefts).astype('uint8')
            new_rights = np.asarray(new_rights).astype('uint8')
            new_labels = np.asarray(new_labels).astype('float32')
            self.debugger.viewTripples(lefts, rights, labels, off=0, how_many=3)
            self.debugger.viewTripples(new_lefts, new_rights, new_labels, off=0, how_many=3)
            """


        if self.settings.verbose >= 3:
            print("Last balance check:")
            self.check_balance_of_data(labels, labels_paths)

        data = [lefts, rights, labels]
        paths = [lefts_paths, rights_paths, labels_paths]
        return data, paths


    def load_dataset_ONLY_PATHS_UPDATE_FROM_THE_OTHER_ONE_IF_NEEDED(self):
        load_paths_from_folders = False  # TRUE To recompute the paths from folder

        if load_paths_from_folders:
            # Load paths
            print("\nLoading all paths from input folders:")
            lefts_paths, rights_paths, labels_paths = self.load_paths_from_folders()
            self.dataLoader.save_paths(lefts_paths, self.settings.large_file_folder + "saved_paths_2012_"+self.dataset_version+"ALL.pickle")
            self.dataLoader.save_paths(rights_paths, self.settings.large_file_folder + "saved_paths_2015_"+self.dataset_version+"ALL.pickle")
            self.dataLoader.save_paths(labels_paths, self.settings.large_file_folder + "saved_paths_vectors_"+self.dataset_version+"ALL.pickle")
        else:
            lefts_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "saved_paths_2012_"+self.dataset_version+"ALL.pickle")
            rights_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "saved_paths_2015_"+self.dataset_version+"ALL.pickle")
            labels_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "saved_paths_vectors_"+self.dataset_version+"ALL.pickle")

        print("We have", len(lefts_paths), "2012 images, ", lefts_paths[0:4])
        print("We have", len(rights_paths), "2015 images, ", rights_paths[0:4])
        print("We have", len(labels_paths), "vector images, ", labels_paths[0:4])

        # Load images
        lefts_paths = lefts_paths[0:self.SUBSET]
        rights_paths = rights_paths[0:self.SUBSET]
        labels_paths = labels_paths[0:self.SUBSET]

        paths = [lefts_paths, rights_paths, labels_paths]
        return paths

    ### Loading file paths manually :

    def load_paths_from_folders(self):

        if self.variant == "256" or self.variant == "256_clean" or self.variant == "256_cleanManual":
            # 256x256 version

            paths_2012 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip1_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip2_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip3_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip4_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip5_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip6_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip7_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip8_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip9_256x256_over32_png/"]

            paths_2015 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip1_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip2_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip3_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip4_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip5_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip6_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip7_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip8_256x256_over32_png/",
                          "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip9_256x256_over32_png/"]

            # paths_vectors = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip7/","/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip8/"]
            # paths_2012 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip7_256x256_over32_png/","/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip8_256x256_over32_png/"]
            # paths_2015 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip7_256x256_over32_png/","/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip8_256x256_over32_png/"]
        if self.variant == "256":
            # 256x256 version

            paths_vectors = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip1/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip2/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip3/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip4/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip5/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip6/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip7/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip8/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip9/"]

            paths_vectors = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip5/"]
        elif self.variant == "256_clean":
            # 256x256 version

            start_dir = "/home/pf/pfstaff/projects/ruzicka/dataset_initial_view(secondInst)/03_06_vectorStrips_areaOnlyBiggerThan40/vector_strips_a40p_256x256_32over/"
            paths_vectors = [   "vector_strip_1_a40p_256_32over/",
                                "vector_strip_2_a40p_256_32over/",
                                "vector_strip_3_a40p_256_32over/",
                                "vector_strip_4_a40p_256_32over/",
                                "vector_strip_5_a40p_256_32over/",
                                "vector_strip_6_a40p_256_32over/",
                                "vector_strip_7_a40p_256_32over/",
                                "vector_strip_8_a40p_256_32over/",
                                "vector_strip_9_a40p_256_32over/"]
            paths_vectors = [start_dir + f for f in paths_vectors]

        elif self.variant == "256_cleanManual":
            # 256x256 version

            start_dir = "/home/pf/pfstaff/projects/ruzicka/CleanedVectors_manually_256x256_32over/"
            paths_vectors = [   "vector_strip1_256x256_over32/",
                                "vector_strip2_256x256_over32/",
                                "vector_strip3_256x256_over32/",
                                "vector_strip4_256x256_over32/",
                                "vector_strip5_256x256_over32/",
                                "vector_strip6_256x256_over32/",
                                "vector_strip7_256x256_over32/",
                                "vector_strip8_256x256_over32/",
                                "vector_strip9_256x256_over32/"]

            paths_vectors = [start_dir + f for f in paths_vectors]


        if self.variant == "112" or self.variant == "112_clean":
            # 112x112 version

            paths_2012 =   ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip1_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip2_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip3_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip4_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip5_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip6_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip7_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip8_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip9_112x112_png/",]

            paths_2015 = [
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip1_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip2_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip3_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip4_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip5_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip6_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip7_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip8_112x112_png/",
                            "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip9_112x112_png/"]

        if  self.variant == "112":
            # 112x112 version
            paths_vectors = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip1/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip2/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip3/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip4/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip5/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip6/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip7/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip8/",
                             "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip9/"
                             ]
        elif self.variant == "112_clean":
            start_dir = "/home/pf/pfstaff/projects/ruzicka/dataset_initial_view(secondInst)/03_06_vectorStrips_areaOnlyBiggerThan40/vector_strips_a40p_112x112/"
            paths_vectors = [   "vector_strip_1_a40p/",
                                "vector_strip_2_a40p/",
                                "vector_strip_3_a40p/",
                                "vector_strip_4_a40p/",
                                "vector_strip_5_a40p/",
                                "vector_strip_6_a40p/",
                                "vector_strip_7_a40p/",
                                "vector_strip_8_a40p/",
                                "vector_strip_9_a40p/"]
            paths_vectors = [start_dir + f for f in paths_vectors]

        if  self.variant == "6368_special":
            paths_2012 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_6368x6368px_large/2012_strip2_6368tiles/"]
            paths_2015 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_6368x6368px_large/2015_strip2_6368tiles/"]
            paths_vectors = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_6368x6368px_large/2015_strip2_6368tiles/"] # hax



        files_paths_2012 = self.load_path_lists(paths_2012)
        all_2012_png_paths, edge_tile_2012, total_tiles_2012 = self.process_path_lists(files_paths_2012, paths_2012)
        files_paths_2015 = self.load_path_lists(paths_2015)
        all_2015_png_paths, _, _ = self.process_path_lists(files_paths_2015, paths_2015)

        files_vectors = self.load_path_lists(paths_vectors)
        all_vector_paths = self.process_path_lists_for_vectors(files_vectors, paths_vectors, edge_tile_2012, total_tiles_2012)

        return all_2012_png_paths, all_2015_png_paths, all_vector_paths

    ### Loading images:
    def check_shapes(self, lefts, rights, labels, lefts_paths, rights_paths, labels_paths):
        # check sizes!
        shit_list = []  # you don't want to get on that list
        for idx in range(len(lefts)):
            left = lefts[idx]
            right = rights[idx]
            label = labels[idx]

            if (left.shape[0] != self.default_raster_shape[0] or left.shape[1] != self.default_raster_shape[1] or
                    left.shape[2] != self.default_raster_shape[2]):
                shit_list.append(idx)
            elif (right.shape[0] != self.default_raster_shape[0] or right.shape[1] != self.default_raster_shape[1] or
                    right.shape[2] != self.default_raster_shape[2]):
                    shit_list.append(idx)
            if (label.shape[0] != self.default_vector_shape[0] or label.shape[1] != self.default_vector_shape[1]):
                shit_list.append(idx)
        off = 0
        for i in shit_list:
            idx = i - off
            print("deleting", idx, lefts[idx].shape, rights[idx].shape, labels[idx].shape)
            #print("deleting", idx, labels[idx].shape)
            #self.debugger.viewTripples([lefts[idx]], [rights[idx]], [labels[idx]], off=0, how_many=1)

            del lefts[idx]
            del rights[idx]
            del labels[idx]
            del lefts_paths[idx]
            del rights_paths[idx]
            del labels_paths[idx]
            off += 1

        return lefts, rights, labels, lefts_paths, rights_paths, labels_paths

    def load_vector_image(self, filename):
        if filename == None:
            arr = np.zeros((self.IMAGE_RESOLUTION,self.IMAGE_RESOLUTION), dtype=float)
            return arr

        img = io.imread(filename)
        arr = np.asarray(img)

        # threshold it
        if self.variant == "256_cleanManual":
            ## FOR NEWER DATASETS
            arr[arr <= 0] = 0
            arr[arr == 65535] = 0 # hi ArcGis ghosts
            arr[arr != 0] = 1

        else:
            ## FOR OLDER DATASETS
            thr = 0
            arr[arr > thr] = 1
            arr[arr <= thr] = 0

            # anything <= 0 (usually just one value) => 0 (no change)
            # anything >  0                          => 1 (change)
        return arr

    def load_raster_image(self, filename):
        img = io.imread(filename)
        arr = np.asarray(img)
        return arr


    def sort_nicely(self, l):
        """ Sort the given list in the way that humans expect.
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        l.sort(key=alphanum_key)

    def get_last_line(self, file):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            return (last_line)

    def load_path_lists(self, paths):
        lists = []
        for path in paths:
            files = []
            for f in tqdm(listdir(path)):
                #if isfile(join(path, f)): # useless and slow!
                files.append(f)

            #files = [f for f in listdir(path) if isfile(join(path, f))]
            self.sort_nicely(files)
            files = [path+f for f in files] # add the full path

            lists.append(files)
        return lists

    def find_edge_tile(self, pgw_files):
        # All these lists contain a rectangle of tiles stored in one array. We can find the "edge" tile of this rectangle
        # by inspecting their location, which is saved in PGw file.
        edge_tile = -1

        rowlocation = self.get_last_line(pgw_files[0])
        for idx, pgw_file in enumerate(pgw_files):
            last_line_rowlocation = self.get_last_line(pgw_file)
            if last_line_rowlocation != rowlocation:
                #print(last_line_rowlocation, "is not", rowlocation, "at", idx, pgw_file)
                edge_tile = idx - 1
                break

        return edge_tile

    def skip_rows_columns(self,png_files, skip_rows,skip_columns,rows, columns):
        selected_indices = []
        #print("Skipping", skip_rows, "row and", skip_columns, "column")
        for row in range(skip_rows, rows - skip_rows):
            for column in range(skip_columns, columns - skip_columns):
                idx = int(row * columns + column)
                selected_indices.append(idx)
        selected_files = [png_files[idx] for idx in selected_indices]
        return selected_files

    def process_path_lists(self, files_paths, folder_paths, ext_file="PNG", ext_geo_file="PGw"):
        all_png_paths = []
        edges_tile_forVEC = []
        total_tiles_forVEC = []
        for i in range(len(folder_paths)):
            files_path = files_paths[i]
            folder_path = folder_paths[i]
            foldername = folder_path.split("/")[-2]
            print("---[",foldername,"]---")

            png_files = [f for f in files_path if f[-3:] == ext_file]
            pgw_files = [f for f in files_path if f[-3:] == ext_geo_file]

            edge_tile = self.find_edge_tile(pgw_files)

            total_tiles = len(png_files)
            columns = edge_tile + 1  # count the 0 too

            if columns == 0:
                columns = 1

            rows = int((total_tiles) / (columns))

            print("We have", columns, "columns x", rows, "rows = ", (columns * rows))

            # By default we skip the sides of each image strip (to prevent "badly behaving" images)
            selected_files = self.skip_rows_columns(png_files, self.local_setting_skip_rows, self.local_setting_skip_columns, rows, columns)

            print("From [",foldername,"] we selected", len(selected_files), "files (we ommited", (total_tiles - len(selected_files)),
                  "files from the sides)")
            all_png_paths += selected_files
            edges_tile_forVEC.append(edge_tile)
            total_tiles_forVEC.append(total_tiles)

        return all_png_paths, edges_tile_forVEC, total_tiles_forVEC


    def process_path_lists_for_vectors(self, files_paths, folder_paths, raster_edge_tile, raster_total_tiles):
        # Vector renders may have files missing, which means these are empty!\
        # let's keep None in there for now so that mutual indexing between the tripples works
        # (a little bit messy)

        all_vec_paths = []
        for i in range(len(folder_paths)):
            files_path = files_paths[i]
            folder_path = folder_paths[i]
            total_tiles = raster_total_tiles[i]
            edge_tile = raster_edge_tile[i]

            foldername = folder_path.split("/")[-2]
            print("---[",foldername,"]---")

            vec_files = [f for f in files_path if f[-3:] == "TIF"]

            tile_if_exists = [None] * total_tiles

            for i in range(len(vec_files)):
                file_string = vec_files[i].split(".TIF")[0]
                num = file_string.split("_")[-1]
                try:
                    tile_if_exists[int(num)] = vec_files[i]
                except:
                    # ignore these, there are some images in vectors which overlap from strip7 into strip8
                    a=42

            columns = edge_tile + 1  # count the 0 too
            if columns == 0:
                columns = 1 # special case with small data
            rows = int((total_tiles) / (columns))

            print("We have", columns, "columns x", rows, "rows = ", (columns * rows))

            # By default we skip the sides of each image strip (to prevent "badly behaving" images)
            selected_files = self.skip_rows_columns(tile_if_exists, self.local_setting_skip_rows, self.local_setting_skip_columns, rows, columns)

            print("From [",foldername,"] we selected", len(selected_files), "files (we ommited", (total_tiles - len(selected_files)),
                  "files from the sides)")
            all_vec_paths += selected_files
        return all_vec_paths

    # Data checking:

    def check_balance_of_data(self, labels, optional_paths=''):
        # In this we want to check how many pixels are marking "change" in each image

        exploration_sum_values = {}
        array_of_number_of_change_pixels = []

        for image in tqdm(labels):
            number_of_ones = np.count_nonzero(image.flatten()) # << loading takes care of this 0 vs non-zero
            array_of_number_of_change_pixels.append(number_of_ones)

        #print("In the whole dataset, we have these values:")
        #print(exploration_sum_values)

        #print("We have these numbers of alive pixels:")
        #print(array_of_number_of_change_pixels)

        self.debugger.save_arr(array_of_number_of_change_pixels)

        # << skip it, if you can
        array_of_number_of_change_pixels = self.debugger.load_arr()

        array_of_number_of_change_pixels = array_of_number_of_change_pixels / (self.IMAGE_RESOLUTION*self.IMAGE_RESOLUTION) * 100.0 # percentage of image changed


        idx_examples_bigger = np.argwhere(array_of_number_of_change_pixels > self.bigger_than_percent)
        original_array_of_number_of_change_pixels = array_of_number_of_change_pixels

        less = [val for val in array_of_number_of_change_pixels if val <= self.bigger_than_percent]
        array_of_number_of_change_pixels = [val for val in array_of_number_of_change_pixels if val > self.bigger_than_percent]
        print("The data which is >",self.bigger_than_percent,"% changed is ", len(array_of_number_of_change_pixels), "versus the remainder of", len(less))

        # the histogram of the data
        #fig = plt.figure(figsize=(10, 8))
        fig = plt.figure()
        bins = 100
        values_of_bins, bins, patches = plt.hist(array_of_number_of_change_pixels, bins, facecolor='g', alpha=0.75)

        print("values_of_bins", np.asarray(values_of_bins).astype(int))
        print("bins sizes", bins)
        plt.yscale('log', nonposy='clip')

        plt.title('How much change in the '+str(self.IMAGE_RESOLUTION)+'x'+str(self.IMAGE_RESOLUTION)+' tiles?')
        plt.xlabel('Percentage of pixels belonging to change')
        plt.ylabel('Log scale of number of images/'+str(self.IMAGE_RESOLUTION)+'x'+str(self.IMAGE_RESOLUTION)+' tiles')

        plt.show()

        if optional_paths is not '':
            labels_to_show = []
            txt_labels = []
            for i in range(np.min([100,len(idx_examples_bigger)-1])):
                idx = idx_examples_bigger[i][0]
                label_image = optional_paths[idx]
                labels_to_show.append(label_image)
                txt_labels.append(original_array_of_number_of_change_pixels[idx])

            import DatasetInstance_OurAerial
            images = [DatasetInstance_OurAerial.DatasetInstance_OurAerial.load_vector_image(0, path) for path in labels_to_show]

            self.debugger.viewVectors(images, txt_labels, how_many=6, off=0)
            #self.debugger.viewVectors(images, txt_labels, how_many=6, off=6)
            #self.debugger.viewVectors(images, txt_labels, how_many=6, off=12)

    def balance_data(self, labels, lefts_paths, rights_paths, labels_paths):
        array_of_number_of_change_pixels = []

        for image in tqdm(labels):
            number_of_ones = np.count_nonzero(image.flatten()) # << loading takes care of this 0 vs non-zero
            array_of_number_of_change_pixels.append(number_of_ones)

        #print("We have these numbers of alive pixels:")
        #print(array_of_number_of_change_pixels)

        self.debugger.save_arr(array_of_number_of_change_pixels, "BALANCING")
        array_of_number_of_change_pixels = self.debugger.load_arr("BALANCING")

        array_of_number_of_change_pixels = array_of_number_of_change_pixels / (
                    self.IMAGE_RESOLUTION * self.IMAGE_RESOLUTION) * 100.0  # percentage of image changed

        idx_examples_bigger = np.argwhere(array_of_number_of_change_pixels > self.bigger_than_percent)

        idx_examples_smaller = np.argwhere(array_of_number_of_change_pixels <= self.smaller_than_percent)

        # build new one mixing two IDX arrays
        print("Mixing from", len(idx_examples_bigger), "bigger and", len(idx_examples_smaller), "smaller.")

        new_lefts_paths = []
        new_rights_paths = []
        new_labels_paths = []

        # all of the "bigger"
        # as many from the "smaller"
        to_select = len(idx_examples_bigger)

        indices_smaller = random.sample(range(0, len(idx_examples_smaller)), to_select) # "NO CHANGE" samples are shuffled across the whole map
        #print(indices_smaller)

        non_repetition = []

        for i in range(0, len(indices_smaller)):
            idx_smaller = idx_examples_smaller[ indices_smaller[i] ][0]
            idx_bigger = idx_examples_bigger[i][0]

            if idx_bigger in non_repetition:
                print(idx_bigger, "already loaded!")
                assert False
            if idx_smaller in non_repetition:
                print(idx_smaller, "already loaded!")
                assert False
            non_repetition.append(idx_bigger)
            non_repetition.append(idx_smaller)

            #print("adding", idx_smaller, idx_bigger)

            new_lefts_paths.append(lefts_paths[idx_smaller])
            new_lefts_paths.append(lefts_paths[idx_bigger])

            new_rights_paths.append(rights_paths[idx_smaller])
            new_rights_paths.append(rights_paths[idx_bigger])

            new_labels_paths.append(labels_paths[idx_smaller])
            new_labels_paths.append(labels_paths[idx_bigger])

        return new_lefts_paths, new_rights_paths, new_labels_paths

    def mask_label_into_class_label(self, mask_labels):
        """
        Converts the mask label images (for example 224x224 pixel image with 0s and 1s) into a single class label
        ("change" or "no change") using the same threshold as when balancing the data.
        PS: we could use different threshold here ...
        Slight problem is that we won't be exactly sure that the "change" is really "change" and not just noisy
        mask label (to do: clean label data)

        :param mask_labels:
        :return:
        """
        array_of_number_of_change_pixels = []

        for mask in tqdm(mask_labels):
            number_of_ones = np.count_nonzero(mask.flatten()) # << loading takes care of this 0 vs non-zero
            array_of_number_of_change_pixels.append(number_of_ones)

        self.debugger.save_arr(array_of_number_of_change_pixels, "BALANCING")
        array_of_number_of_change_pixels = self.debugger.load_arr("BALANCING")

        array_of_number_of_change_pixels = array_of_number_of_change_pixels / (
                    self.IMAGE_RESOLUTION * self.IMAGE_RESOLUTION) * 100.0  # percentage of image changed

        class_labels = []
        for value in array_of_number_of_change_pixels:
            is_change = value > self.bigger_than_percent
            class_labels.append(int(is_change))

        return np.array(class_labels)
