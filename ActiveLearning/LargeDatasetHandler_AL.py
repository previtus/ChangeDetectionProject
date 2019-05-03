import numpy as np
from tqdm import tqdm
import ActiveLearning.HelperFunctions as Helpers

from random import sample

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

    Todo:
    - data aug (behind the indices)
    - shuffle indices (each time after an epoch...)
    - add indices, remove indices (support for AL)

    Todo outside, bigger picture
    - AL outline, have two sets - TrainSet and UnlabeledSet and move data in between them... (always everything we have?)

    """

    def __init__(self, settings, create_inmemory_or_ondemand = "ondemand"):
        self.settings = settings

        self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND = create_inmemory_or_ondemand
        #self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND = "inmemory" # or "ondemand"
        self.data_in_memory = {}
        self.labels_in_memory = {}

        self.N_of_data = None
        self.indices = None # Array of indices, doesn't have to be sorted
        self.paths = [{},{},{}] # these should also be dictionaries so that you get path from the idx
        self.dataaug_descriptors = {}

    def report(self):
        lefts_paths, rights_paths, labels_paths = self.paths

        print("[LargeDataset] contains:")
        print("\t[\paths] left:", len(lefts_paths), ", right:", len(rights_paths), ", labels:", len(labels_paths))
        print("\t[\data] in memory loaded:", len(self.data_in_memory))
        print("\t[\labels] in memory loaded:", len(self.labels_in_memory))

    def get_number_of_samples(self):
        return self.N_of_data

    def initialize_from_just_paths(self, paths):
        # in this case we are creating it for the first time, from just an array of paths
        lefts_paths, rights_paths, labels_paths = paths
        lefts_paths_dictionary = {}
        rights_paths_dictionary = {}
        labels_paths_dictionary = {}
        self.dataaug_descriptors = {}

        N = len(lefts_paths)
        assert N == len(rights_paths)
        assert N == len(labels_paths) # assert we have all paths loaded and available

        self.N_of_data = N
        self.indices = range(N)

        for idx in self.indices:
            lefts_paths_dictionary[idx] = lefts_paths[idx]
            rights_paths_dictionary[idx] = rights_paths[idx]
            labels_paths_dictionary[idx] = labels_paths[idx]
            self.dataaug_descriptors[idx] = 0 # no dataaug happened

        self.paths = lefts_paths_dictionary, rights_paths_dictionary, labels_paths_dictionary

        return 0

    """
    def initialize_from_another_object(self, data_in_memory, labels_in_memory, N_of_data, indices, paths, dataaug_descriptors, KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND_flag):
        self.data_in_memory = data_in_memory
        self.labels_in_memory = labels_in_memory
        self.N_of_data = N_of_data
        self.indices = indices
        self.paths = paths
        self.dataaug_descriptors = dataaug_descriptors
        self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND = KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND_flag
    """


    def keep_it_all_in_memory(self, optional_h5_path = None):
        self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND = "inmemory"

        if optional_h5_path:
            self.data_in_memory, self.labels_in_memory = self.load_images_into_memory_from_h5_file(optional_h5_path)
        else:
            all_paths = self.paths
            self.data_in_memory, self.labels_in_memory = self.load_images_into_memory(all_paths)


    # LOADERs ==========================================================================================================

    def load_images_into_memory(self, paths, skip_labels=False):
        # paths keeps three dictionaries!
        #self.data_in_memory[ IDX ] = corresponding data - Left and Right
        #self.labels_in_memory[ IDX ] = corresponding label - Label
        data_in_memory = {}
        labels_in_memory = {}
        lefts_paths, rights_paths, labels_paths = paths
        #print("lefts_paths.keys()", lefts_paths.keys())

        print("\nLoading whole set of raster images and labels:")
        lefts = {}
        for idx in tqdm(lefts_paths):
            path = lefts_paths[idx]
            lefts[idx] = Helpers.load_raster_image(path)
        rights = {}
        for idx in tqdm(rights_paths):
            path = rights_paths[idx]
            rights[idx] = Helpers.load_raster_image(path)
        labels = {}
        if not skip_labels:
            for idx in tqdm(labels_paths):
                path = labels_paths[idx]
                labels[idx] = Helpers.load_vector_image(path)

        for idx in lefts.keys():
            data_in_memory[idx] = [lefts[idx], rights[idx]]
        if not skip_labels:
            for idx in labels.keys():
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

        lefts_paths = {}
        rights_paths = {}
        labels_paths = {}

        for idx in selected_indices:
            lefts_paths[idx] = self.paths[0][idx]
            rights_paths[idx] = self.paths[1][idx]
            labels_paths[idx] = self.paths[2][idx]

        selected_paths = lefts_paths, rights_paths, labels_paths

        return selected_paths

    def load_images_by_indices(self, selected_indices, skip_labels=False):
        # returns some subset of images of the dataset
        if self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "inmemory":
            selected_data = {}
            selected_labels = {}
            for idx in selected_indices:
                selected_data[idx] = self.data_in_memory[idx]
                selected_labels[idx] = self.labels_in_memory[idx]
            #selected_data = [self.data_in_memory[i] for i in selected_indices]
            #selected_labels = [self.labels_in_memory[i] for i in selected_indices]
            return selected_data, selected_labels # dicts
        else:
            selected_paths = self.paths_by_indices(selected_indices)
            data_in_memory, labels_in_memory = self.load_images_into_memory(selected_paths, skip_labels=skip_labels) # dicts
            return data_in_memory, labels_in_memory

    # Data generator ===================================================================================================

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
                    for i in data_in_memory.keys():
                        #for i in range(len(data_in_memory)):
                        lefts.append(data_in_memory[i][0])
                        rights.append(data_in_memory[i][1])

                    data_batch = selected_indices, [lefts,rights]

                elif mode == 'datalabels':
                    data_in_memory, labels_in_memory = self.load_images_by_indices(selected_indices)
                    # X is as [lefts, rights]

                    lefts = []
                    rights = []
                    for i in data_in_memory.keys():
                        #for i in range(len(data_in_memory)):
                        lefts.append(data_in_memory[i][0])
                        rights.append(data_in_memory[i][1])
                    labels = labels_in_memory
                    data_batch = selected_indices, [lefts, rights], labels

                yield data_batch

            LOOPING = 0 # keras fit_generator might need this on 1 ... then the limitation would be done with steps_per_epoch set to <int_loop_times>

    def get_all_data_as_arrays(self):
        # makes sense to fit the preprocessor for example ... (if that doesn't fit into memory anymore, we can do some
        # tricks instead)

        # dictionaries to arrays
        lefts_paths, rights_paths, labels_paths = self.paths

        if self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "ondemand":
            print("loading the data from files again (if this happens more than once per epoch for each dataset, then it's inefficient!")
            data_in_memory, labels_in_memory = self.load_images_into_memory(self.paths)
        else:
            data_in_memory, labels_in_memory = self.data_in_memory, self.labels_in_memory

        lefts = []
        rights = []
        labels = []

        #print("self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND", self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND)
        #print("data_in_memory len", len(data_in_memory))
        #print("labels_in_memory len", len(labels_in_memory))

        for idx in self.indices:
            left = data_in_memory[idx][0]
            right = data_in_memory[idx][1]
            label = labels_in_memory[idx]

            lefts.append(left)
            rights.append(right)
            labels.append(label)

        #data_in_memory[idx] = [lefts[idx], rights[idx]]
        #labels_in_memory[idx] = labels[idx]

        lefts = np.asarray(lefts).astype('float32')
        rights = np.asarray(rights).astype('float32')
        labels = np.asarray(labels).astype('float32') # labels weren't really changed in here

        data = [lefts, rights, labels]
        paths = [lefts_paths, rights_paths, labels_paths]

        return data, paths

    # Sampling =========================================================================================================

    def sample_random_indice_subset(self, how_many):
        if len(self.indices) < how_many:
            print("Trying to sample more than what we have in the dataset! Will cause an error...")
        selected_indices = sample(self.indices, how_many) # random sampling without replacement
        return selected_indices

    def sample_random_indice_subset_balanced_classes(self, how_many):
        print("STILL NEEDS TO BE IMPLEMENTED")
        assert False


    # Dataset splitting ================================================================================================

    # Todo> fuctions:
    # - get_items_by_indices() > gives everything in the dataset under these indices
    # - add_items() < allows to add new items to a dataset (if the new data is not loaded and we are adding to loaded
    #                   dataset, do load it ... on the opposite case, deleted the actual data)
    # - remove_by_indice()


    def pop_items(self, indices_to_pop):
        # return as dictionaries (which can be added to a new object)
        # while deleting them from this one


        popped_data_in_memory = {}
        popped_labels_in_memory = {}
        popped_dataaug_descriptors = {}
        popped_paths_lefts = {}
        popped_paths_rights = {}
        popped_paths_labels = {}

        for idx in indices_to_pop:
            # pop it from current structures, add it to the new ones

            if self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "inmemory":
                popped_data_in_memory[idx] = self.data_in_memory.pop(idx)
                popped_labels_in_memory[idx] = self.labels_in_memory.pop(idx)

            popped_dataaug_descriptors[idx] = self.dataaug_descriptors.pop(idx)
            popped_paths_lefts[idx] = self.paths[0].pop(idx)
            popped_paths_rights[idx] = self.paths[1].pop(idx)
            popped_paths_labels[idx] = self.paths[2].pop(idx)

        popped_paths = popped_paths_lefts, popped_paths_rights, popped_paths_labels

        new_indices = [idx for idx in self.indices if idx not in indices_to_pop]
        new_N_of_data = len(new_indices)
        self.indices = new_indices
        self.N_of_data = new_N_of_data

        mem_flag = self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND

        packed = popped_data_in_memory, popped_labels_in_memory, popped_dataaug_descriptors, popped_paths, indices_to_pop, mem_flag

        return packed

    def add_items(self, packed_items):
        added_data_in_memory, added_labels_in_memory, added_dataaug_descriptors, added_paths, indices_to_added, mem_flag = packed_items
        added_paths_lefts, added_paths_rights, added_paths_labels = added_paths

        if self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "inmemory" and mem_flag == "ondemand":
            print("Will have to load them to memory!")

            added_data_in_memory, added_labels_in_memory = self.load_images_into_memory(added_paths)

        elif self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "ondemand" and mem_flag == "inmemory":
            print("Will have to delete them from memory!")
            for idx in indices_to_added:
                added_data_in_memory.pop(idx)
                added_labels_in_memory.pop(idx)

        for idx in indices_to_added:

            if self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "inmemory":
                self.data_in_memory[idx] = added_data_in_memory.pop(idx)
                self.labels_in_memory[idx] = added_labels_in_memory.pop(idx)

            self.dataaug_descriptors[idx] = added_dataaug_descriptors.pop(idx)

            self.paths[0][idx] = added_paths_lefts.pop(idx)
            self.paths[1][idx] = added_paths_rights.pop(idx)
            self.paths[2][idx] = added_paths_labels.pop(idx)

        self.indices = list(self.paths[0].keys())
        self.N_of_data = len(self.indices)

    """
    def remove_indices(self, remove_indices):

        new_indices = [idx for idx in self.indices if idx not in remove_indices]
        new_N_of_data = len(new_indices)


        # dictionaries:
        new_data_in_memory = {}
        new_labels_in_memory = {}
        new_paths = [{},{},{}]
        new_dataaug_descriptors = {}

        # removed the items from the dictionaries

    def add_items(self, items):
        print(FOO)

    def split_dataset_into_two(self, indices_to_move_to_another_set):
        # add them to the new sets and remove them from old ones ...

        items = self.select_by_indices()

        SubDataset = LargeDatasetHandler_AL(self.settings)
        SubDataset.add_items(items)
        #SubDataset.initialize_from_another_object(new_data_in_memory, new_labels_in_memory, new_N_of_data, new_indices, new_paths,
        #                                          new_dataaug_descriptors, self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND)

        self.remove_indices(indices_to_move_to_another_set)

        return 0
    """

def tmp_get_whole_dataset(in_memory=False):
    from ActiveLearning.LargeDatasetHandler_AL import LargeDatasetHandler_AL
    import Settings

    # init structures
    import mock
    args = mock.Mock()
    args.name = "test"


    settings = Settings.Settings(args)
    WholeDataset = LargeDatasetHandler_AL(settings)

    # load paths of our favourite dataset!
    import DataLoader, DataPreprocesser, Debugger
    import DatasetInstance_OurAerial

    dataLoader = DataLoader.DataLoader(settings)
    debugger = Debugger.Debugger(settings)

    h5_file = settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL3.0_1.0_sel2144_res256x256.h5"

    datasetInstance = DatasetInstance_OurAerial.DatasetInstance_OurAerial(settings, dataLoader, "256_cleanManual")

    # ! this one automatically balances the data + deletes misfits in the resolution
    data, paths = datasetInstance.load_dataset()
    lefts_paths, rights_paths, labels_paths = paths
    print("Paths: L,R,Y ", len(lefts_paths), len(rights_paths), len(labels_paths))

    # ! this one loads them all (CHECK: would some be deleted?)
    # paths = datasetInstance.load_dataset_ONLY_PATHS_UPDATE_FROM_THE_OTHER_ONE_IF_NEEDED()
    # lefts_paths, rights_paths, labels_paths = paths
    # print("Paths: L,R,Y ", len(lefts_paths), len(rights_paths), len(labels_paths))

    WholeDataset.initialize_from_just_paths(paths)

    if in_memory:
        #WholeDataset.keep_it_all_in_memory()
        WholeDataset.keep_it_all_in_memory(h5_file)

    WholeDataset.report()

    return WholeDataset