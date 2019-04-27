import numpy as np
from tqdm import tqdm
import ActiveLearning.HelperFunctions as Helpers

FOO = None

class LargeDatasetHandler_AL(object):
    """
    Large dataset handler. Designed to work with Active Learning.
    Should have separate indices (and descriptors for data augmentation if need be) and the actual data.
    Image data can be loaded on demand (we are predicting that it wouldn't potentially fit into memory.

    Q: can we save it as tfrecords? And have the on demand loading just touching the file?

    Plan:

    Init from the dataset - load all indices (and paths) and descriptors for augmentation
    (not the labels or images right now, but available on demand).

    Make another instance of the class where we select just a subset (initial Training set, which will gradually grow
    over iterations) - and also actually keep the RemainingUnlabeled set.
    It should be easy to remove some indices from one and add them to another (gradually move samples from RemainingUnlabeled to Training).

    Should be easy to train a model (or an ensemble of models) on it's data.

    Can the memory threshold be automatic (eventually)?

    """

    def __init__(self, settings):
        self.settings = settings

        self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND = "ondemand"
        #self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND = "inmemory" # or "ondemand"
        self.data_in_memory = {}
        self.labels_in_memory = {}

        self.N_of_data = None
        self.indices = None
        self.paths = [[],[],[]]
        self.dataaug_descriptors = None

    def report(self):
        lefts_paths, rights_paths, labels_paths = self.paths

        print("[LargeDataset] contains:")
        print("\t[\paths] left:", len(lefts_paths), ", right:", len(rights_paths), ", labels:", len(labels_paths))
        print("\t[\data] in memory loaded:", len(self.data_in_memory))
        print("\t[\labels] in memory loaded:", len(self.labels_in_memory))

    def initialize_from_just_paths(self, paths):
        # in this case we are creating it for the first time, from just an array of paths
        lefts_paths, rights_paths, labels_paths = paths

        N = len(lefts_paths)
        assert N == len(rights_paths)
        assert N == len(labels_paths) # assert we have all paths loaded and available

        self.N_of_data = N
        self.indices = range(N)
        self.paths = paths

        self.dataaug_descriptors = np.zeros(N)

        return 0


    def keep_it_all_in_memory(self, optional_h5_path = None):
        self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND = "inmemory"

        if optional_h5_path:
            self.data_in_memory, self.labels_in_memory = self.load_images_into_memory_from_h5_file(optional_h5_path)
        else:
            all_paths = self.paths
            self.data_in_memory, self.labels_in_memory = self.load_images_into_memory(all_paths)


    # LOADERs ==========================================================================================================

    def load_images_into_memory(self, paths, skip_labels=False):
        #self.data_in_memory[ IDX ] = corresponding data - Left and Right
        #self.labels_in_memory[ IDX ] = corresponding label - Label
        data_in_memory = {}
        labels_in_memory = {}
        lefts_paths, rights_paths, labels_paths = paths

        print("\nLoading whole set of raster images and labels:")
        lefts = []
        for path in tqdm(lefts_paths):
            lefts.append(Helpers.load_raster_image(path))
        rights = []
        for path in tqdm(rights_paths):
            rights.append(Helpers.load_raster_image(path))
        labels = []
        if not skip_labels:
            for path in tqdm(labels_paths):
                labels.append(Helpers.load_vector_image(path))

        for idx in range(len(lefts)):
            data_in_memory[idx] = [lefts[idx], rights[idx]]
        if not skip_labels:
            for idx in range(len(labels)):
                labels_in_memory[idx] = labels[idx]

        return data_in_memory, labels_in_memory

    def load_images_into_memory_from_h5_file(self, hdf5_path):
        #self.data_in_memory[ IDX ] = corresponding data - Left and Right
        #self.labels_in_memory[ IDX ] = corresponding label - Label
        data_in_memory = {}
        labels_in_memory = {}
        print("loading images from:", hdf5_path)

        lefts, rights, labels = Helpers.load_images_from_h5(hdf5_path)
        lefts = np.asarray(lefts).astype('uint8')
        rights = np.asarray(rights).astype('uint8')
        labels = np.asarray(labels).astype('float32')

        for idx in range(len(lefts)):
            data_in_memory[idx] = lefts[idx], rights[idx]
        for idx in range(len(labels)):
            labels_in_memory[idx] = labels[idx]

        return data_in_memory, labels_in_memory

    def paths_by_indices(self, selected_indices):
        # paths have left,rights and labels in it
        lefts_paths = [self.paths[0][i] for i in selected_indices]
        rights_paths = [self.paths[1][i] for i in selected_indices]
        labels_paths = [self.paths[2][i] for i in selected_indices]
        selected_paths = lefts_paths, rights_paths, labels_paths

        return selected_paths

    def load_images_by_indices(self, selected_indices, skip_labels=False):
        # returns some subset of images of the dataset
        if self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "inmemory":
            selected_data = [self.data_in_memory[i] for i in selected_indices]
            selected_labels = [self.labels_in_memory[i] for i in selected_indices]
            return selected_data, selected_labels
        else:
            selected_paths = self.paths_by_indices(selected_indices)
            data_in_memory, labels_in_memory = self.load_images_into_memory(selected_paths, skip_labels=skip_labels)
            return data_in_memory, labels_in_memory


    # mode > indices, dataonly, datalabels
    def generator_for_all_images(self, BATCH_SIZE=2048, mode='indices'):
        # over time also returns all the images of the dataset, goes in the batches
        # see: https://github.com/keras-team/keras/issues/107
        LOOPING = 1
        while LOOPING: # repeat forever
            loop_times = self.N_of_data / BATCH_SIZE
            int_loop_times = int(np.floor(loop_times)) + 1
            # +1 => last batch will be with less samples (1224 instead of the full 2048)

            print("Dataset loop finished (loops ", loop_times, "with this generator), that is (int):", int_loop_times)
            for i in range(int_loop_times):
                #if i % 125 == 0:
                #    print("i = " + str(i))
                end_of_selection = (i + 1) * BATCH_SIZE
                end_of_selection = min(end_of_selection, len(self.indices))
                selected_indices = list(self.indices[i * BATCH_SIZE:end_of_selection])
                print(selected_indices)

                if mode == 'indices':
                    data_batch = [selected_indices] # INDICES

                elif mode == 'dataonly':
                    data_in_memory, _ = self.load_images_by_indices(selected_indices, skip_labels=True)
                    # X is as [lefts, rights]
                    lefts = []
                    rights = []
                    for i in range(len(data_in_memory)):
                        lefts.append(data_in_memory[i][0])
                        rights.append(data_in_memory[i][1])

                    data_batch = selected_indices, [lefts,rights]

                elif mode == 'datalabels':
                    data_in_memory, labels_in_memory = self.load_images_by_indices(selected_indices)
                    # X is as [lefts, rights]

                    lefts = []
                    rights = []
                    for i in range(len(data_in_memory)):
                        lefts.append(data_in_memory[i][0])
                        rights.append(data_in_memory[i][1])
                    labels = labels_in_memory
                    data_batch = selected_indices, [lefts, rights], labels

                yield data_batch

            LOOPING = 0 # keras fit_generator might need this on 1 ... then the limitation would be done with steps_per_epoch set to <int_loop_times>