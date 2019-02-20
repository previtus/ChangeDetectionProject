import Debugger

import os
from os import listdir
from os.path import isfile, join
import re
import pickle
import numpy as np
from skimage import io
from tqdm import tqdm
import h5py


class DataLoader(object):
    """
    Will handle loading and parsing the data.
    """


    def __init__(self, settings):
        self.settings = settings
        self.debugger = Debugger.Debugger(settings)


    ### h5 save and load
    def mkdir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_images_to_h5(self, lefts, rights, labels, savepath):

        SIZE = lefts[0].shape
        SUBSET = len(lefts)
        self.mkdir(self.settings.large_file_folder+"datasets")
        hdf5_path = self.settings.large_file_folder+"datasets/"+savepath + str(SUBSET) + "_res" + str(SIZE[0]) + "x" + str(SIZE[1]) + ".h5"

        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("lefts", data=lefts, dtype="float32")
        hdf5_file.create_dataset("rights", data=rights, dtype="float32")
        hdf5_file.create_dataset("labels", data=labels, dtype="float32")
        hdf5_file.close()

        print("Saved", SUBSET, "images successfully to:", hdf5_path)

        return hdf5_path

    def load_images_from_h5(self, hdf5_path):
        hdf5_file = h5py.File(hdf5_path, "r")
        lefts = hdf5_file['lefts'][:]
        rights = hdf5_file['rights'][:]
        labels = hdf5_file['labels'][:]
        hdf5_file.close()

        return lefts, rights, labels

    ### Loading file paths from already computed files:

    def load_paths_from_pickle(self, name):
        with open(name, 'rb') as fp:
            paths = pickle.load(fp)
        return paths

    def save_paths(self, paths, name):
        with open(name, 'wb') as fp:
            pickle.dump(paths, fp)


