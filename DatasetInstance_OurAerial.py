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


    def __init__(self, settings, dataLoader, variant = 256):
        self.settings = settings
        self.dataLoader = dataLoader
        self.debugger = Debugger.Debugger(settings)

        self.variant = variant # 256 = 256x256, 112 = 112x112

        self.local_setting_skip_rows = 2
        self.local_setting_skip_columns = 2

        self.save_path_ = "OurAerial_preloadedImgs_sub"

        if self.variant == 256:
            self.dataset_version = "256x256_over32"
            self.SUBSET = 83000
            self.SUBSET = 5000
            self.SUBSET = -1
            self.IMAGE_RESOLUTION = 256

            self.bigger_than_percent = 3.0  # 8.0 from full set
            self.smaller_than_percent = 0.0  # 3.0 ?

            self.default_raster_shape = (256,256,4)
            self.default_vector_shape = (256,256)

            # select one manually! sub<X> says how many actual samples it has (after balancing and valid checks)
            self.hdf5_path = self.settings.large_file_folder + "datasets/" + self.save_path_ + "5000_res256x256.h5"
            #self.hdf5_path = self.settings.large_file_folder + "datasets/" + self.save_path_ + "1000_res256x256.h5"
            self.hdf5_path = self.settings.large_file_folder + "datasets/OurAerial_preloadedImgs_sub708_res256x256.h5"

        elif  self.variant == 112:
            self.dataset_version = "112x112"
            #self.SUBSET = 118667
            self.SUBSET = 1000
            self.SUBSET = 5000
            self.SUBSET = -1
            self.SUBSET = 80000

            self.IMAGE_RESOLUTION = 112

            self.bigger_than_percent = 18.0  # 18.0
            self.smaller_than_percent = 0.0  # 5.0

            self.default_raster_shape = (112, 112, 4)
            self.default_vector_shape = (112, 112)

            # select one manually! sub<X> says how many actual samples it has (after balancing and valid checks)
            self.hdf5_path = self.settings.large_file_folder + "datasets/" + self.save_path_ + "5000_res112x112.h5"
            #self.hdf5_path = self.settings.large_file_folder + "datasets/" + self.save_path_ + "1000_res112x112.h5"
            self.hdf5_path = self.settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL2.0_0.0_sel18_res112x112.h5"



    def get_paths(self):
        lefts_paths = []
        rights_paths = []
        labels_paths = []
        return lefts_paths, rights_paths, labels_paths

    #def datasetSpecificEdit_rasters(self,data):
    #    return data
    #def datasetSpecificEdit_vectors(self,data):
    #    return data

    def present_thyself(self):
        print("Our own dataset of aerial photos. Resolution goes in the variants of 256x256x4 and 112x112x4 (channels: near infra, r,g,b).")


    def load_dataset(self):
        load_paths_from_folders = False  # TRUE To recompute the paths from folder
        load_images_anew = True         # TRUE To reload images from the files directly + rebalance them

        # load_image_paths()
        # save_image_paths_to_cache()

        if load_paths_from_folders:
            # Load paths
            print("Loading all paths from input folders:")
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

        # check_valid_images() + balance_images()
        # save_valid_and_balanced_paths_to_cache()

        if load_images_anew:
            # Load data

            # It would be more elegant:
            # 1 load labels
            # 2 check balance
            # 3 load from these
            # 4 check shapes

            print("Loading vector images:")
            labels = []
            for path in tqdm(labels_paths):
                labels.append(self.load_vector_image(path))
            lefts = []
            for path in lefts_paths:
                lefts.append([[0]])  # empty now
            rights = []
            for path in rights_paths:
                rights.append([[0]]) # empty now

            # balance data
            print("Checking the balance in labels:")
            lefts, rights, labels, lefts_paths, rights_paths, labels_paths = self.balance_data(lefts, rights, labels, lefts_paths, rights_paths, labels_paths)

            print("Loading balanced set of raster images:")
            new_lefts = []
            for path in tqdm(lefts_paths):
                new_lefts.append(self.load_raster_image(path))
            lefts = new_lefts
            new_rights = []
            for path in tqdm(rights_paths):
                new_rights.append(self.load_raster_image(path))
            rights = new_rights

            lefts, rights, labels, lefts_paths, rights_paths, labels_paths = self.check_shapes(lefts, rights, labels, lefts_paths, rights_paths, labels_paths)


            #self.check_balance_of_data(labels, labels_paths)  ## CHECKING HOW MUCH CHANGE OCCURED, SLOW

            labels = np.asarray(labels).astype('float32')
            rights = np.asarray(rights).astype('float32')
            lefts = np.asarray(lefts).astype('float32')

            # Save
            self.dataLoader.save_images_to_h5(lefts, rights, labels, self.save_path_+"BAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+"_sel")
            lefts = np.asarray(lefts).astype('uint8')
            rights = np.asarray(rights).astype('uint8')

            self.dataLoader.save_paths(lefts_paths, self.settings.large_file_folder + "saved_paths_2012_"+self.dataset_version+"BALVAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+".pickle")
            self.dataLoader.save_paths(rights_paths, self.settings.large_file_folder + "saved_paths_2015_"+self.dataset_version+"BALVAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+".pickle")
            self.dataLoader.save_paths(labels_paths, self.settings.large_file_folder + "saved_paths_vectors_"+self.dataset_version+"BALVAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+".pickle")

        else:
            # These loaded images are valid (all the same resolution) and balanced according to the setting.

            lefts, rights, labels = self.dataLoader.load_images_from_h5(self.hdf5_path)
            lefts = np.asarray(lefts).astype('uint8')
            rights = np.asarray(rights).astype('uint8')
            labels = np.asarray(labels).astype('float32')

            lefts_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "saved_paths_2012_"+self.dataset_version+"BALVAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+".pickle")
            rights_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "saved_paths_2015_"+self.dataset_version+"BALVAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+".pickle")
            labels_paths = self.dataLoader.load_paths_from_pickle(
                self.settings.large_file_folder + "saved_paths_vectors_"+self.dataset_version+"BALVAL"+str(self.bigger_than_percent)+"_"+str(self.smaller_than_percent)+".pickle")


        # self.debugger.viewTripples(lefts, rights, labels, off=0, how_many=3)
        data = [lefts, rights, labels]
        paths = [lefts_paths, rights_paths, labels_paths]
        return data, paths

    ### Loading file paths manually :

    def load_paths_from_folders(self):

        if self.variant == 256:
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


        elif  self.variant == 112:
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
        # anything <= 0 (usually just one value) => 0 (no change)
        # anything >  0                          => 1 (change)

        thr = 0
        arr[arr > thr] = 1
        arr[arr <= thr] = 0
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
                if isfile(join(path, f)):
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

        #labels = labels[0:500]

        exploration_sum_values = {}
        array_of_number_of_change_pixels = []

        # slow part >>
        for image in tqdm(labels):
            values = self.debugger.occurancesInImage(image) # values we have are "1" and "0.0"
            #print("values.keys()",values.keys())
            for value in values:
                #print("'",value,"'")
                if value in exploration_sum_values:
                    exploration_sum_values[value] += values[value]
                else:
                    exploration_sum_values[value] = values[value]
            if 1 in values.keys(): # number 1 as a key signifies changed pixel
                array_of_number_of_change_pixels.append(values[1])
            else:
                array_of_number_of_change_pixels.append(0)

        print("In the whole dataset, we have these values:")
        print(exploration_sum_values)

        print("We have these numbers of alive pixels:")
        print(array_of_number_of_change_pixels)

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


    def balance_data(self, lefts, rights, labels, lefts_paths, rights_paths, labels_paths):
        # SLOW!
        array_of_number_of_change_pixels = []

        for image in tqdm(labels):
            values = self.debugger.occurancesInImage(image) # values we have are "1" and "0.0"
            if 1 in values.keys(): # number 1 as a key signifies changed pixel
                array_of_number_of_change_pixels.append(values[1])
            else:
                array_of_number_of_change_pixels.append(0)

        print("We have these numbers of alive pixels:")
        print(array_of_number_of_change_pixels)

        self.debugger.save_arr(array_of_number_of_change_pixels, "BALANCING")

        # Pre-Computed by the helper
        array_of_number_of_change_pixels = self.debugger.load_arr("BALANCING")

        array_of_number_of_change_pixels = array_of_number_of_change_pixels / (
                    self.IMAGE_RESOLUTION * self.IMAGE_RESOLUTION) * 100.0  # percentage of image changed

        idx_examples_bigger = np.argwhere(array_of_number_of_change_pixels > self.bigger_than_percent)

        idx_examples_smaller = np.argwhere(array_of_number_of_change_pixels <= self.smaller_than_percent)

        # build new one mixing two IDX arrays
        print("Mixing from", len(idx_examples_bigger), "bigger and", len(idx_examples_smaller), "smaller.")

        new_lefts = []
        new_rights = []
        new_labels = []
        new_lefts_paths = []
        new_rights_paths = []
        new_labels_paths = []

        # all of the "bigger"
        # as many from the "smaller"
        to_select = len(idx_examples_bigger)

        indices_smaller = random.sample(range(0, len(idx_examples_smaller)), to_select)
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

            new_lefts.append(lefts[idx_smaller])
            new_lefts.append(lefts[idx_bigger])

            new_rights.append(rights[idx_smaller])
            new_rights.append(rights[idx_bigger])

            new_labels.append(labels[idx_smaller])
            new_labels.append(labels[idx_bigger])

            new_lefts_paths.append(lefts_paths[idx_smaller])
            new_lefts_paths.append(lefts_paths[idx_bigger])

            new_rights_paths.append(rights_paths[idx_smaller])
            new_rights_paths.append(rights_paths[idx_bigger])

            new_labels_paths.append(labels_paths[idx_smaller])
            new_labels_paths.append(labels_paths[idx_bigger])

        return new_lefts, new_rights, new_labels, new_lefts_paths, new_rights_paths, new_labels_paths
