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

        self.local_setting_skip_rows = 2
        self.local_setting_skip_columns = 2
        self.IMAGE_RESOLUTION = settings.default_raster_shape[0]

        self.dataset = self.load_dataset()
        # Alternation!
        #self.dataset = self.alt_dataset_onera_satellite_change_detection()


    def load_dataset(self):
        load_paths_from_folders = False # TRUE To recompute the paths from folder
        load_images_anew = True # TRUE To reload images from the files directly
        
        hdf5_path = self.settings.large_file_folder+"datasets/preloadedImgs_sub5000_res256x256.h5"
        hdf5_path = self.settings.large_file_folder+"datasets/preloadedImgs_sub1000_res256x256.h5"

        dataset_version = "_112x112" # or "_256x256_over32"

        if load_paths_from_folders:
            # Load paths
            all_2012_png_paths, all_2015_png_paths, all_vector_paths = self.load_paths_from_folders()
            self.save_paths(all_2012_png_paths, self.settings.large_file_folder+"saved_paths_2012_112x112.pickle")
            self.save_paths(all_2015_png_paths, self.settings.large_file_folder+"saved_paths_2015_112x112.pickle")
            self.save_paths(all_vector_paths, self.settings.large_file_folder+"saved_paths_vectors_112x112.pickle")
        else:
            all_2012_png_paths = self.load_paths_from_pickle(self.settings.large_file_folder+"saved_paths_2012_112x112.pickle")
            all_2015_png_paths = self.load_paths_from_pickle(self.settings.large_file_folder+"saved_paths_2015_112x112.pickle")
            all_vector_paths = self.load_paths_from_pickle(self.settings.large_file_folder+"saved_paths_vectors_112x112.pickle")

        print("We have",len(all_2012_png_paths), "2012 images, ", all_2012_png_paths[0:4])
        print("We have",len(all_2015_png_paths), "2015 images, ", all_2015_png_paths[0:4])
        print("We have",len(all_vector_paths), "vector images, ", all_vector_paths[0:4])

        # Load images
        SUBSETa = 0
        SUBSETb = 118667
        all_2012_png_paths = all_2012_png_paths[SUBSETa:SUBSETb]
        all_2015_png_paths = all_2015_png_paths[SUBSETa:SUBSETb]
        all_vector_paths = all_vector_paths[SUBSETa:SUBSETb]


        if load_images_anew:
            # Load data
            lefts = []
            for path in tqdm( all_2012_png_paths ):
                #lefts.append(self.load_raster_image(path))
                lefts.append([[0]]) # hax
            rights = []
            for path in tqdm( all_2015_png_paths ):
                #rights.append(self.load_raster_image(path))
                rights.append([[0]]) # hax
            labels = []
            for path in tqdm( all_vector_paths ):
                #labels.append(self.load_vector_image(path))
                labels.append([[0]]) # hax

            #lefts, rights, labels = self.check_shapes(lefts, rights, labels)
            self.debugger.check_balance_of_data(labels, all_vector_paths) ## CHECKING HOW MUCH CHANGE OCCURED, SLOW

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
            #< HAX
            #if (left.shape[0] != self.settings.default_raster_shape[0] or left.shape[1] != self.settings.default_raster_shape[1] or
            #        left.shape[2] != self.settings.default_raster_shape[2]):
            #    shit_list.append(idx)
            #elif (right.shape[0] != self.settings.default_raster_shape[0] or right.shape[1] != self.settings.default_raster_shape[1] or
            #        right.shape[2] != self.settings.default_raster_shape[2]):
            #        shit_list.append(idx)
            if (label.shape[0] != self.settings.default_vector_shape[0] or label.shape[1] != self.settings.default_vector_shape[1]):
                shit_list.append(idx)
        off = 0
        for i in shit_list:
            idx = i - off
            # < HAX
            #print("deleting", idx, lefts[idx].shape, rights[idx].shape, labels[idx].shape)
            print("deleting", idx, labels[idx].shape)
            # < HAX
            #self.debugger.viewTripples([lefts[idx]], [rights[idx]], [labels[idx]], off=0, how_many=1)

            del lefts[idx]
            del rights[idx]
            del labels[idx]
            off += 1

        return lefts, rights, labels

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

    ### Loading file paths from already computed files:

    def load_paths_from_pickle(self, name):
        with open(name, 'rb') as fp:
            paths = pickle.load(fp)
        return paths

    ### Loading file paths manually :

    def load_paths_from_folders(self):

        # 112x112 version
        paths_vectors = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip1/",
                        "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip2/",
                        "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/vectorLabels/vector_strip3/"]

        paths_2012 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip1_112x112_png/",
                        "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip2_112x112_png/",
                        "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2012_strip3_112x112_png/"]

        paths_2015 = ["/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip1_112x112_png/",
                        "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip2_112x112_png/",
                        "/home/pf/pfstaff/projects/ruzicka/TiledDataset_112x112/2015_strip3_112x112_png/"]

        # 256x256 version
        """
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
        """
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


    ### Alternative dataset, the "Onera Satellite Change Detection dataset"
    def alt_dataset_onera_satellite_change_detection(self):
        lefts = []
        rights = []
        labels = []

        # Mess

        images_path = "/scratch/ruzicka/dataset_initial_view/02_11_OSCD_dataset/63_dataset/Onera Satellite Change Detection dataset - Images/"
        labels_path = "/scratch/ruzicka/dataset_initial_view/02_11_OSCD_dataset/63_training/Onera Satellite Change Detection dataset - Train Labels/"
        train = "aguasclaras,bercy,bordeaux,nantes,paris,rennes,saclay_e,abudhabi,cupertino,pisa,beihai,hongkong,beirut,mumbai"
        test = "brasilia,montpellier,norcia,rio,saclay_w,valencia,dubai,lasvegas,milano,chongqing"

        # for each in train:
        # image: /<name>/pair/img1.png x img2.png
        # label: /<name>/cm/cm.png

        trains = train.split(",")
        tests = test.split(",")

        train_x_left_paths = []
        train_x_right_paths = []
        train_y_label_paths = []

        for train in trains:
            path = images_path + train + "/pair/"
            train_x_left_paths.append(path + "img1.png")
            train_x_right_paths.append(path + "img2.png")
            # train_y_label_paths.append(labels_path + train + "/cm/cm.png")
            train_y_label_paths.append(labels_path + train + "/cm/" + train + "-cm.tif")
        # tile_paths_old = [f for f in listdir(path_old) if isfile(join(path_old, f))]

        test_x_left_paths = []
        test_x_right_paths = []

        for test in tests:
            path = images_path + test + "/pair/"
            test_x_left_paths.append(path + "img1.png")
            test_x_right_paths.append(path + "img2.png")

        print(len(train_x_left_paths), train_x_left_paths)
        print(len(train_x_right_paths), train_x_right_paths)
        print(len(train_y_label_paths), train_y_label_paths)

        # load as images and possibly also crop!
        # tile it into 112x112 tiles?
        from PIL import Image

        def tile_from_image(img, desired_size):
            repetitions = int(np.floor(min(img.size[0] / desired_size, img.size[1] / desired_size)))

            min_size = min(img.size[0], img.size[1])
            # print(desired_size, "fits", repetitions, "times into", min_size, "(", desired_size*repetitions, ")")

            width, height = img.size
            left = np.ceil((width - min_size) / 2.)
            top = np.ceil((height - min_size) / 2.)
            right = width - np.floor((width - min_size) / 2)
            bottom = height - np.floor((height - min_size) / 2)

            cropped_im = img.crop((left, top, right, bottom))

            # print(cropped_im.size)

            resize_to = desired_size * repetitions

            resized_im = cropped_im.resize((resize_to, resize_to))
            # print(resized_im.size)

            # resized_im.show()
            tiles = []
            # now crop it into repetitions*repetitions times!
            for x_off_i in range(repetitions):
                for y_off_i in range(repetitions):
                    left = x_off_i * desired_size
                    top = y_off_i * desired_size
                    right = left + desired_size
                    bottom = top + desired_size

                    box = (left, top, right, bottom)
                    # print(box)
                    segment = resized_im.crop(box)
                    # print(segment.size)
                    # segment.show()
                    tiles.append(segment)
            return tiles, resized_im

        desired_size = 112
        train_X_left = []
        train_X_right = []
        train_Y = []

        for i in range(len(train_x_left_paths)):
            # for i in range(1):
            train_left_img = Image.open(train_x_left_paths[i])
            train_right_img = Image.open(train_x_right_paths[i])
            train_label_img = Image.open(train_y_label_paths[i])

            train_label_img = train_label_img.convert('L')

            left_images, _ = tile_from_image(train_left_img, desired_size)
            right_images, _ = tile_from_image(train_right_img, desired_size)
            label_images, _ = tile_from_image(train_label_img, desired_size)

            # label_rebuilt = rebuild_image_from_tiles(label_images)

            # label_rebuilt.show()

            train_X_left += left_images
            train_X_right += right_images
            train_Y += label_images

        print("loaded", len(train_X_left), len(train_X_right), len(train_Y), "train images")

        train_X_left = [np.asarray(i) for i in train_X_left]
        train_X_right = [np.asarray(i) for i in train_X_right]
        train_Y = [np.asarray(i) for i in train_Y]

        train_X_left = np.asarray(train_X_left)
        train_X_right = np.asarray(train_X_right)
        train_Y = np.asarray(train_Y)

        def explore_set_stats(arr_set):
            amin = np.amin(arr_set.flatten())
            amax = np.amax(arr_set.flatten())
            print("min", amin, "max", amax, " ... avg", np.mean(arr_set.flatten()), "+-", np.std(arr_set.flatten()))

        #train_X_left = train_X_left / 255.0
        #train_X_right = train_X_right / 255.0
        # labels are in 1 to 2 spread -> send them to 0-1
        train_Y = (train_Y - 1.0)

        print("left images")
        explore_set_stats(train_X_left)
        print("right images")
        explore_set_stats(train_X_right)
        print("label images")
        explore_set_stats(train_Y)

        print(train_X_left[0].shape)
        print("shapes")
        print(train_Y[0].shape)
        print(train_Y.shape)

        #train_Y = train_Y.reshape(train_Y.shape + (1,)) # happens later
        # train_Y = train_Y.reshape((383, 112, 112, 1))

        print(train_Y[0].shape)

        ############### VIZ 1 ###################################################
        if False:
            fig = plt.figure(figsize=(8, 8))
            columns = 3  # needs to be multiple of 3
            rows = 3
            # off = random.randint(1,1001)
            for i in range(1, columns * rows + 1, 3):
                # idx = i * 20 + off
                idx = i * 2 + random.randint(1, 100)

                # img_old = mpimg.imread(path_old + tile_paths_old[idx])
                img_old = train_X_left[idx]
                fig.add_subplot(rows, columns, i)
                plt.imshow(img_old)
                fig.gca().set_axis_off()

                # img_new = mpimg.imread(path_new + tile_paths_new[idx])
                img_new = train_X_right[idx]
                fig.add_subplot(rows, columns, i + 1)
                plt.imshow(img_new)
                fig.gca().set_axis_off()

                img_lab = train_Y[idx]
                fig.add_subplot(rows, columns, i + 2)
                plt.imshow(img_lab[:, :, 0])
                fig.gca().set_axis_off()

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            fig.suptitle("Showing the same space in two points in time")

            fig.text(0.5, 0.04, 'Showing pairs: left, right', ha='center')

            plt.show()
        # =============================================================================

        lefts = train_X_left
        rights = train_X_right
        labels = train_Y

        dataset = [lefts, rights, labels]
        return dataset
