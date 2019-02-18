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
        self.debugger = Debugger.Debugger()

        self.local_setting_skip_rows = 2
        self.local_setting_skip_columns = 2

        self.dataset = self.load_dataset()


    def load_dataset(self):
        load_paths_from_folders = True # TRUE To recompute the paths from folder
        load_images_anew = False # TRUE To reload images from the files directly
        
        hdf5_path = self.settings.large_file_folder+"datasets/preloadedImgs_sub5000_res256x256.h5"
        hdf5_path = self.settings.large_file_folder+"datasets/preloadedImgs_sub1000_res256x256.h5"

        if load_paths_from_folders:
            # Load paths
            all_2012_png_paths, all_2015_png_paths, all_vector_paths = self.load_paths_from_folders()
            self.save_paths(all_2012_png_paths, self.settings.large_file_folder+"saved_paths_2012.pickle")
            self.save_paths(all_2015_png_paths, self.settings.large_file_folder+"saved_paths_2015.pickle")
            self.save_paths(all_vector_paths, self.settings.large_file_folder+"saved_paths_vectors.pickle")
        else:
            all_2012_png_paths = self.load_paths_from_pickle(self.settings.large_file_folder+"saved_paths_2012.pickle")
            all_2015_png_paths = self.load_paths_from_pickle(self.settings.large_file_folder+"saved_paths_2015.pickle")
            all_vector_paths = self.load_paths_from_pickle(self.settings.large_file_folder+"saved_paths_vectors.pickle")

        print("We have",len(all_2012_png_paths), "2012 images, ", all_2012_png_paths[0:4])
        print("We have",len(all_2015_png_paths), "2015 images, ", all_2015_png_paths[0:4])
        print("We have",len(all_vector_paths), "vector images, ", all_vector_paths[0:4])
        """
        # Load images
        SUBSETa = 0
        SUBSETb = 1000
        all_2012_png_paths = all_2012_png_paths[SUBSETa:SUBSETb]
        all_2015_png_paths = all_2015_png_paths[SUBSETa:SUBSETb]
        all_vector_paths = all_vector_paths[SUBSETa:SUBSETb]
        """

        if load_images_anew:
            # Load data
            lefts = []
            for path in tqdm( all_2012_png_paths ):
                lefts.append(self.load_raster_image(path))
            rights = []
            for path in tqdm( all_2015_png_paths ):
                rights.append(self.load_raster_image(path))
            labels = []
            for path in tqdm( all_vector_paths ):
                labels.append(self.load_vector_image(path))

            lefts, rights, labels = self.check_shapes(lefts, rights, labels)

            labels = np.asarray(labels).astype('float32')
            rights = np.asarray(rights).astype('float32')
            lefts = np.asarray(lefts).astype('float32')

            # Save
            self.save_images_to_h5(lefts, rights, labels)
            lefts = np.asarray(lefts).astype('uint8')
            rights = np.asarray(rights).astype('uint8')

        else:
            lefts, rights, labels = self.load_images_from_h5(hdf5_path)
            lefts = np.asarray(lefts).astype('uint8')
            rights = np.asarray(rights).astype('uint8')
            labels = np.asarray(labels).astype('float32')

        #self.debugger.viewTripples(lefts, rights, labels, off=0, how_many=3)
        dataset = [lefts, rights, labels]
        return dataset

    ### h5 save and load
    def mkdir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_images_to_h5(self, lefts, rights, labels):

        SIZE = lefts[0].shape
        SUBSET = len(lefts)
        self.mkdir(self.settings.large_file_folder+"datasets")
        hdf5_path = self.settings.large_file_folder+"datasets/preloadedImgs_sub" + str(SUBSET) + "_res" + str(SIZE[0]) + "x" + str(SIZE[1]) + ".h5"

        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("lefts", data=lefts, dtype="float32")
        hdf5_file.create_dataset("rights", data=rights, dtype="float32")
        hdf5_file.create_dataset("labels", data=labels, dtype="float32")
        hdf5_file.close()

        print("Saved", SUBSET, "images successfully to:", hdf5_path)

    def load_images_from_h5(self, hdf5_path):
        hdf5_file = h5py.File(hdf5_path, "r")
        lefts = hdf5_file['lefts'][:]
        rights = hdf5_file['rights'][:]
        labels = hdf5_file['labels'][:]
        hdf5_file.close()

        return lefts, rights, labels

    ### Loading images:
    def check_shapes(self, lefts, rights, labels):
        # check sizes!
        shit_list = []  # you don't want to get on that list
        for idx in range(len(lefts)):
            left = lefts[idx]
            right = rights[idx]
            label = labels[idx]
            if (left.shape[0] != self.settings.default_raster_shape[0] or left.shape[1] != self.settings.default_raster_shape[1] or
                    left.shape[2] != self.settings.default_raster_shape[2]):
                shit_list.append(idx)
            elif (right.shape[0] != self.settings.default_raster_shape[0] or right.shape[1] != self.settings.default_raster_shape[1] or
                    right.shape[2] != self.settings.default_raster_shape[2]):
                    shit_list.append(idx)
            elif (label.shape[0] != self.settings.default_vector_shape[0] or label.shape[1] != self.settings.default_vector_shape[1]):
                shit_list.append(idx)
        off = 0
        for i in shit_list:
            idx = i - off
            print("deleting", idx, lefts[idx].shape, rights[idx].shape, labels[idx].shape)
            self.debugger.viewTripples([lefts[idx]], [rights[idx]], [labels[idx]], off=0, how_many=1)

            del lefts[idx]
            del rights[idx]
            del labels[idx]
            off += 1

        return lefts, rights, labels

    def load_vector_image(self, filename):
        if filename == None:
            arr = np.zeros((256,256), dtype=float)
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

    ### Loading file paths from already computed files:

    def load_paths_from_pickle(self, name):
        with open(name, 'rb') as fp:
            paths = pickle.load(fp)
        return paths

    ### Loading file paths manually :

    def load_paths_from_folders(self):
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

        #paths_vectors = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip7/","/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/vectorLabels/strip8/"]
        #paths_2012 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip7_256x256_over32_png/","/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip8_256x256_over32_png/"]
        #paths_2015 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip7_256x256_over32_png/","/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip8_256x256_over32_png/"]

        files_paths_2012 = self.load_path_lists(paths_2012)
        all_2012_png_paths, edge_tile_2012, total_tiles_2012 = self.process_path_lists(files_paths_2012, paths_2012)
        files_paths_2015 = self.load_path_lists(paths_2015)
        all_2015_png_paths, _, _ = self.process_path_lists(files_paths_2015, paths_2015)

        files_vectors = self.load_path_lists(paths_vectors)
        all_vector_paths = self.process_path_lists_for_vectors(files_vectors, paths_vectors, edge_tile_2012, total_tiles_2012)

        return all_2012_png_paths, all_2015_png_paths, all_vector_paths

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

    def save_paths(self, paths, name):
        with open(name, 'wb') as fp:
            pickle.dump(paths, fp)

