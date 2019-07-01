import numpy as np
from tqdm import tqdm
import ActiveLearning.HelperFunctions as Helpers
import Debugger

from random import sample

FOO = None
BATCH_PRECOMP_FOLDER = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/entire_dataset_as_batch_chunks/"

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
        self.original_indices = None # Array of indices as they were in their original order (which coincidentally was range(N))
                                     # will be used as a reference to which indices have been removed ...
                                     # Used only with the "RemainingUnlabeledSet" (which is never added items, just slowly poped)
        self.paths = [{},{},{}] # these should also be dictionaries so that you get path from the idx
        self.dataaug_descriptors = {}

        self.per_tile_class = {} # class "change" or "no-change" - in some cases we will precompute these!
        self.has_per_tile_class_computed = False

        # for balance stats
        self.debugger = Debugger.Debugger(settings)

    # Getters / Setters
    def report(self):
        lefts_paths, rights_paths, labels_paths = self.paths

        print("[LargeDataset] contains:")
        print("\t[\paths] left:", len(lefts_paths), ", right:", len(rights_paths), ", labels:", len(labels_paths))
        print("\t[\data] in memory loaded:", len(self.data_in_memory))
        print("\t[\labels] in memory loaded:", len(self.labels_in_memory))
        if self.has_per_tile_class_computed:
            print("\t[\label classes] in memory loaded:", len(self.per_tile_class))
            self.report_balance_from_tile_classes()

    def get_number_of_samples(self):
        return self.N_of_data

    def report_balance_of_class_labels(self, DEBUG_GET_IDX=False):
        # we are interested in how many samples from "change" vs. "non-change" we have in our dataset

        # for the one dataset we are using .....
        bigger_than_percent = 3.0
        smaller_than_percent = 1.0
        IMAGE_RESOLUTION = 256


        ## < Reused
        array_of_number_of_change_pixels = []
        for idx in self.labels_in_memory:
            label_image = self.labels_in_memory[idx]
            number_of_ones = np.count_nonzero(label_image.flatten()) # << loading takes care of this 0 vs non-zero
            array_of_number_of_change_pixels.append(number_of_ones)

        array_of_number_of_change_pixels = np.asarray(array_of_number_of_change_pixels)

        array_of_number_of_change_pixels = array_of_number_of_change_pixels / (
                    IMAGE_RESOLUTION * IMAGE_RESOLUTION) * 100.0  # percentage of image changed

        idx_examples_bigger = np.argwhere(array_of_number_of_change_pixels > bigger_than_percent)
        idx_examples_smaller = np.argwhere(array_of_number_of_change_pixels <= smaller_than_percent)
        ## >

        N_change_class = len(idx_examples_bigger)
        N_notchange_class = len(idx_examples_smaller)

        # build new one mixing two IDX arrays
        print("We have change : non change in this ratio - ", N_change_class, ":", N_notchange_class, " = ", (N_change_class / N_notchange_class))

        if not DEBUG_GET_IDX:
            return N_change_class, N_notchange_class
        else:
            as_ones_and_zeros = np.where(array_of_number_of_change_pixels > bigger_than_percent,1.0,0.0)
            return N_change_class, N_notchange_class, as_ones_and_zeros, idx_examples_bigger, idx_examples_smaller

    def report_balance_from_tile_classes(self, get_indices = False):
        assert self.has_per_tile_class_computed

        change_indices = []
        nochange_indices = []
        ignored_indices = []

        N_changes = 0 # > 3%
        N_notchanges = 0 # <= 1%
        N_ignored = 0 # in the middle of those

        for key in self.per_tile_class:
            value = self.per_tile_class[key]

            if value == 0.0:
                N_notchanges += 1

                if get_indices:
                    nochange_indices.append(key)
            elif value == 1.0:
                N_changes += 1

                if get_indices:
                    change_indices.append(key)
            elif value == -1.0:
                N_ignored += 1

                if get_indices:
                    ignored_indices.append(key)
            else:
                assert False # error in Matrix!


        if get_indices:
            return change_indices, nochange_indices, ignored_indices
        else:
            print("We have change : non change in this ratio - ", N_changes, ":", N_notchanges, " = ",
                  (N_changes / N_notchanges))
            print("PS: we also have ", N_ignored, "normally ignored samples ... ")

    def classify_label_image(self, image):
        bigger_than_percent = 3.0
        smaller_than_percent = 1.0
        IMAGE_RESOLUTION = 256

        number_of_ones = np.count_nonzero(image.flatten())
        number_of_change_pixels = number_of_ones / (IMAGE_RESOLUTION * IMAGE_RESOLUTION) * 100.0

        if number_of_change_pixels <= smaller_than_percent:
            return 0.0
        if number_of_change_pixels > bigger_than_percent:
            return 1.0
        else:
            # special case of ignored data samples! (if we want to include them we can add them to "no change" as well
            return -1.0

    # Init
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
        self.original_indices = range(N)

        for idx in self.indices:
            lefts_paths_dictionary[idx] = lefts_paths[idx]
            rights_paths_dictionary[idx] = rights_paths[idx]
            labels_paths_dictionary[idx] = labels_paths[idx]
            self.dataaug_descriptors[idx] = 0 # no dataaug happened

        self.paths = lefts_paths_dictionary, rights_paths_dictionary, labels_paths_dictionary

        return 0

    def keep_it_all_in_memory(self, optional_h5_path = None):
        self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND = "inmemory"

        if optional_h5_path:
            self.data_in_memory, self.labels_in_memory = self.load_images_into_memory_from_h5_file(optional_h5_path)
        else:
            all_paths = self.paths
            self.data_in_memory, self.labels_in_memory = self.load_images_into_memory(all_paths)


    # LOADERs ==========================================================================================================

    def load_images_into_memory(self, paths, skip_labels=False, skip_data=False):
        # paths keeps three dictionaries!
        #self.data_in_memory[ IDX ] = corresponding data - Left and Right
        #self.labels_in_memory[ IDX ] = corresponding label - Label
        data_in_memory = {}
        labels_in_memory = {}
        lefts_paths, rights_paths, labels_paths = paths
        #print("lefts_paths.keys()", lefts_paths.keys())

        print("\nLoading whole set of raster images and labels:")
        lefts = {}
        rights = {}
        if not skip_data:
            for idx in tqdm(lefts_paths):
                path = lefts_paths[idx]
                lefts[idx] = Helpers.load_raster_image(path)
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

        """DONE ONCE
        print("DEBUG MESSAGE: RE-SAVED THE h5 FILE AS SMALLER!")
        hdf5_path_lower_size_I_believe = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/datasets/OurAerial_preloadedImgs_subBAL3.0_1.0_sel2144_res256x256_SMALLER.h5"
        Helpers.save_images_to_h5_DEFAULT_DATA_FORMAT(lefts, rights, labels, hdf5_path_lower_size_I_believe)
        """

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

    def load_images_by_indices(self, selected_indices, skip_labels=False, skip_data=False):
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
            data_in_memory, labels_in_memory = self.load_images_into_memory(selected_paths, skip_labels=skip_labels, skip_data=skip_data) # dicts
            return data_in_memory, labels_in_memory

    def load_chunk_of_images_from_h5_without_removed_ones(self, BATCH_ID, BATCH_FOLDER):
        hdf5_path = BATCH_FOLDER + BATCH_ID
        # this leads to a .h5 file

        # Load:
        data_in_memory = {}
        labels_in_memory = {}
        print("loading images from:", hdf5_path)

        lefts, rights, labels, indices = Helpers.load_images_from_h5_with_wholeDataset_indices(hdf5_path)
        lefts = np.asarray(lefts).astype('uint8')
        rights = np.asarray(rights).astype('uint8')
        labels = np.asarray(labels).astype('float32')
        corresponding_indices = np.asarray(indices).astype('int')

        print("Loaded (before removal):", len(lefts), "lefts, ", len(rights), "rights and", len(labels), "labels with",
              len(corresponding_indices), "indices.")

        if self.original_indices is not None:
            print("debug: len(self.original_indices)", len(self.original_indices))
        if self.indices is not None:
            print("debug: len(self.indices)", len(self.indices))

        # we don't care about the removed indices
        # removed indices = all (original) indices - current indices
        removed_indices = [idx for idx in self.original_indices if idx not in self.indices]
        print("debug: len(removed_indices)", len(removed_indices))
        remaining_indices = [idx for idx in corresponding_indices if idx not in removed_indices]
        print("debug: len(remaining_indices)", len(remaining_indices))

        for idx in range(len(lefts)):
            orig_idx = corresponding_indices[idx]
            if orig_idx not in removed_indices:
                data_in_memory[orig_idx] = lefts[idx], rights[idx] # reconstruct them as dictionaries

        for idx in range(len(labels)):
            orig_idx = corresponding_indices[idx]
            if orig_idx not in removed_indices:
                labels_in_memory[orig_idx] = labels[idx]

        print("debug: len(data_in_memory.keys())", len(data_in_memory.keys()))
        print("debug: len(labels_in_memory.keys())", len(labels_in_memory.keys()))

        return data_in_memory, labels_in_memory, remaining_indices

    def compute_per_tile_class_in_batches(self):
        """
        This function is designed with large datasets which won't normally fit in memory in mind - however sometimes we
        still would like to access the labels in some compressed way (such as having a binary label).
        (For example when wanting to sample randomly, but still maintaining a certain label ratio ~ enforced/simulated law of large numbers)

        It will take long time to recompute. Consider saving and reusing a temp file ...
        :return:
        """
        self.per_tile_class = {}

        # goes through all ...
        for batch in self.generator_for_all_images(2048, mode='labelsonly'):  # Yields a large batch sample
            indices = batch[0]
            print("indices from", indices[0], "to", indices[-1])

            labels = batch[1]  # labels in dictionary

            for key in labels:
                label = labels[key]
                #print("key", key, "=", label)

                classification = self.classify_label_image(label)

                self.per_tile_class[key] = classification

            print("in self.per_tile_class",len(self.per_tile_class.keys()))

        self.has_per_tile_class_computed = True

    def save_per_tile_class(self, path='my_file.npy'):
        np.save(path, self.per_tile_class)
    def load_per_tile_class(self, path='my_file.npy'):
        self.per_tile_class = np.load(path).item()
        self.has_per_tile_class_computed = True

    # Data generator ===================================================================================================

    # mode > indices, dataonly, datalabels
    def generator_for_all_images(self, BATCH_SIZE=2048, mode='indices', custom_indices_to_sample_from = None, skip_i_batches = 0,
                                 requested_exactly_these_indices_to_load = None):
        # over time also returns all the images of the dataset, goes in the batches
        # see: https://github.com/keras-team/keras/issues/107
        LOOPING = 1
        while LOOPING: # repeat forever
            if custom_indices_to_sample_from is None:
                loop_times = self.N_of_data / BATCH_SIZE
            else:
                loop_times = len(custom_indices_to_sample_from) / BATCH_SIZE

            if mode == 'dataonly_LOADBATCHFILES':
                # we do have to load all the files and check for samples randomly spread in them ...
                # for that reason the loop_times will be always the max.
                if self.original_indices is not None:
                    loop_times = len(self.original_indices) / BATCH_SIZE
                else:
                    loop_times = 83144 / BATCH_SIZE

            int_loop_times = int(np.floor(loop_times)) + 1
            # +1 => last batch will be with less samples (1224 instead of the full 2048)

            print("Dataset loop finished (loops ", loop_times, "with this generator), that is (int):", int_loop_times)
            for i in range(int_loop_times):
                #if i % 125 == 0:
                #    print("i = " + str(i))
                if custom_indices_to_sample_from is None:
                    end_of_selection = (i + 1) * BATCH_SIZE
                    end_of_selection = min(end_of_selection, len(self.indices))
                    selected_indices = list(self.indices[i * BATCH_SIZE:end_of_selection])
                    #print(selected_indices)
                else:
                    end_of_selection = (i + 1) * BATCH_SIZE
                    end_of_selection = min(end_of_selection, len(custom_indices_to_sample_from))
                    selected_indices = list(custom_indices_to_sample_from[i * BATCH_SIZE:end_of_selection])

                if i < skip_i_batches:
                    print("Skipped batch i=",i)
                    continue
                # 00i_2048_from83144.h5
                BATCH_ID = str(i).zfill(3) + "_" + str(BATCH_SIZE) + "_from" + str(self.N_of_data) + ".h5"
                BATCH_ID = str(i).zfill(3) + "_" + str(BATCH_SIZE) + "_from83144.h5"

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

                elif mode == 'dataonly_LOADBATCHFILES':
                    # In this case we load precomputed h5 files as batch chunks and then remove the items that were already removed from it.
                    # (This might also result in completely empty batches in which case we should just go for another one)

                    data_in_memory, labels_in_memory, corresponding_indices = self.load_chunk_of_images_from_h5_without_removed_ones(BATCH_ID,BATCH_PRECOMP_FOLDER)

                    # X is as [lefts, rights]
                    lefts = []
                    rights = []
                    for i in data_in_memory.keys():
                        #for i in range(len(data_in_memory)):
                        lefts.append(data_in_memory[i][0])
                        rights.append(data_in_memory[i][1])

                    data_batch = corresponding_indices, [lefts,rights] # ps: can contain less items that the requested batch size ...

                elif mode == 'dataonly_LOADBATCHFILES_REQUESTED_INDICES_ONLY':
                    print(requested_exactly_these_indices_to_load)
                    added_data_in_memory = {}
                    added_labels_in_memory = {}
                    received_indices = []

                    data_in_memory, labels_in_memory, corresponding_indices = self.load_chunk_of_images_from_h5_without_removed_ones(BATCH_ID,BATCH_PRECOMP_FOLDER)
                    for idx in data_in_memory.keys():
                        if idx in requested_exactly_these_indices_to_load:
                            added_data_in_memory[idx] = data_in_memory[idx]
                            added_labels_in_memory[idx] = labels_in_memory[idx]
                            received_indices.append(idx)

                    data_batch = added_data_in_memory, added_labels_in_memory, received_indices

                elif mode == 'labelsonly':
                    _, labels_in_memory = self.load_images_by_indices(selected_indices, skip_data=True)
                    data_batch = selected_indices, labels_in_memory

                elif mode == 'datalabels':
                    data_in_memory, labels_in_memory = self.load_images_by_indices(selected_indices)
                    # X is as [lefts, rights]

                    lefts = []
                    rights = []
                    for i in data_in_memory.keys():
                        #for i in range(len(data_in_memory)):
                        lefts.append(data_in_memory[i][0])
                        rights.append(data_in_memory[i][1])

                    labels = []
                    for i in labels_in_memory.keys():
                        labels.append( labels_in_memory[i] )
                    data_batch = selected_indices, [lefts, rights], labels

                elif mode == 'leftpathsandindices':

                    selected_paths_left = {}
                    for idx in selected_indices:
                        print(self.paths[0][idx])
                        selected_paths_left[idx] = self.paths[0][idx]

                    data_batch = [selected_indices, selected_paths_left]

                elif mode == 'SAVEBATCHFILES':
                    print("SAVING RUN ~~~ " + BATCH_ID)
                    print("MAKE SURE THIS IS DONE ONLY WITH THE UNLABANCED DATASET WHEN WE DIDNT REMOVE ANYTHING FROM IT!")
                    # THIS IS A SPECIAL CASE WHICH LEADS TO MAKING'N'BAKIN OF THE NECESSARY BATCHES
                    # Can be later used in the mode "dataonly_LOADBATCHFILES"

                    hdf5_path = BATCH_PRECOMP_FOLDER + BATCH_ID

                    data_in_memory, labels_in_memory = self.load_images_by_indices(selected_indices)
                    lefts = []
                    rights = []
                    for i in data_in_memory.keys():
                        #for i in range(len(data_in_memory)):
                        lefts.append(data_in_memory[i][0])
                        rights.append(data_in_memory[i][1])

                    labels = []
                    for i in labels_in_memory.keys():
                        labels.append( labels_in_memory[i] )

                    print("About to save:", len(lefts), "lefts, ", len(rights), "rights and", len(labels), "labels with", len(selected_indices), "indices.")

                    data_batch = Helpers.save_images_to_h5_with_wholeDataset_indices(lefts, rights, labels, selected_indices, hdf5_path)

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

        return data, paths, self.indices

    # Sampling =========================================================================================================

    def sample_random_indice_subset(self, how_many):
        if len(self.indices) < how_many:
            print("Trying to sample more than what we have in the dataset! Will cause an error...")
        selected_indices = sample(self.indices, how_many) # random sampling without replacement
        return selected_indices

    def sample_random_indice_subset_balanced_classes(self, how_many, ratio = 0.1):
        how_many_nochanges = int(how_many*(1.0-ratio))
        how_many_changes = how_many - how_many_nochanges # always round

        print("Sampling from the dataset in ratio=",ratio," which means",how_many_changes,"changes and",how_many_nochanges, "no-changes.")

        change_indices, nochange_indices, ignored_indices = self.report_balance_from_tile_classes(get_indices=True)
        # get indices of change and no_change class ...

        selected_changes = []
        selected_nochanges = []
        if how_many_changes > 0:
            selected_changes = sample(change_indices, how_many_changes) # random sampling without replacement
        if how_many_nochanges > 0:
            selected_nochanges = sample(nochange_indices, how_many_nochanges) # random sampling without replacement

        # mix these two
        selected_indices = np.append(selected_changes,selected_nochanges)
        np.random.shuffle(selected_indices)

        return selected_indices


    # Dataset splitting ================================================================================================

    # Todo> fuctions:
    # - get_items_by_indices() > gives everything in the dataset under these indices
    # - add_items() < allows to add new items to a dataset (if the new data is not loaded and we are adding to loaded
    #                   dataset, do load it ... on the opposite case, deleted the actual data)
    # - remove_by_indice()

    def get_all_indices_for_saving(self):
        # as a part of saving a loading,
        # if we get all the indices from a set, then we can pop() from entire LargeDatasetHandler without problems
        return self.indices

    def pop_items(self, indices_to_pop):
        # return as dictionaries (which can be added to a new object)
        # while deleting them from this one


        popped_data_in_memory = {}
        popped_labels_in_memory = {}
        popped_dataaug_descriptors = {}
        popped_paths_lefts = {}
        popped_paths_rights = {}
        popped_paths_labels = {}
        popped_per_tile_class = {}

        for idx in indices_to_pop:
            # pop it from current structures, add it to the new ones

            if self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "inmemory":
                popped_data_in_memory[idx] = self.data_in_memory.pop(idx)
                popped_labels_in_memory[idx] = self.labels_in_memory.pop(idx)

            if self.has_per_tile_class_computed:
                popped_per_tile_class[idx] = self.per_tile_class.pop(idx)

            popped_dataaug_descriptors[idx] = self.dataaug_descriptors.pop(idx)
            popped_paths_lefts[idx] = self.paths[0].pop(idx)
            popped_paths_rights[idx] = self.paths[1].pop(idx)
            popped_paths_labels[idx] = self.paths[2].pop(idx)

        popped_paths = popped_paths_lefts, popped_paths_rights, popped_paths_labels

        new_indices = [idx for idx in self.indices if idx not in indices_to_pop]
        # however the self.orignal_indices remains the same

        new_N_of_data = len(new_indices)
        self.indices = new_indices
        self.N_of_data = new_N_of_data

        mem_flag = self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND
        clas_flag = self.has_per_tile_class_computed

        packed = popped_data_in_memory, popped_labels_in_memory, popped_dataaug_descriptors, popped_paths, indices_to_pop, mem_flag, popped_per_tile_class, clas_flag

        return packed

    def add_items(self, packed_items):
        added_data_in_memory, added_labels_in_memory, added_dataaug_descriptors, added_paths, indices_to_added, mem_flag, added_per_tile_class, clas_flag = packed_items
        added_paths_lefts, added_paths_rights, added_paths_labels = added_paths

        if self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "inmemory" and mem_flag == "ondemand":
            print("Will have to load them to memory!")

            if False:
                added_data_in_memory = {}
                added_labels_in_memory = {}
                received_indices = []
                for batch in self.generator_for_all_images(PER_BATCH=2048, mode='dataonly_LOADBATCHFILES_REQUESTED_INDICES_ONLY', requested_exactly_these_indices_to_load=indices_to_added):  # Yields a large batch sample
                    # batch = data_in_memory, labels_in_memory, corresponding_indices
                    batch_added_data_in_memory, batch_added_labels_in_memory, batch_received_indices = batch
                    # double_check(indices_to_added < - > received_indices)
                    # merge batch_* into added_*

            added_data_in_memory, added_labels_in_memory = self.load_images_into_memory(added_paths)

        elif self.KEEP_IT_IN_MEMORY_OR_LOAD_ON_DEMAND == "ondemand" and mem_flag == "inmemory":
            print("Will have to delete them from memory!")
            for idx in indices_to_added:
                added_data_in_memory.pop(idx)
                added_labels_in_memory.pop(idx)

        if clas_flag and self.has_per_tile_class_computed:
            for idx in indices_to_added:
                self.per_tile_class[idx] = added_per_tile_class.pop(idx)
        if not clas_flag and self.has_per_tile_class_computed:
            print("Newly added items don't have class label! re-compute it")
            assert False

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

def get_balanced_dataset(in_memory=False, TMP_WHOLE_UNBALANCED = False):
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

    #h5_file = settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL3.0_1.0_sel2144_res256x256.h5"
    h5_file = settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL3.0_1.0_sel2144_res256x256_SMALLER.h5"

    datasetInstance = DatasetInstance_OurAerial.DatasetInstance_OurAerial(settings, dataLoader, "256_cleanManual")

    if not TMP_WHOLE_UNBALANCED:
        # ! this one automatically balances the data + deletes misfits in the resolution
        data, paths = datasetInstance.load_dataset()
        lefts_paths, rights_paths, labels_paths = paths
        print("Paths: L,R,Y ", len(lefts_paths), len(rights_paths), len(labels_paths))

    else:
        # ! this one loads them all (CHECK: would some be deleted?)
        paths = datasetInstance.load_dataset_ONLY_PATHS_UPDATE_FROM_THE_OTHER_ONE_IF_NEEDED()
        lefts_paths, rights_paths, labels_paths = paths
        print("Paths: L,R,Y ", len(lefts_paths), len(rights_paths), len(labels_paths))

    WholeDataset.initialize_from_just_paths(paths)

    if in_memory:
        assert not TMP_WHOLE_UNBALANCED
        #WholeDataset.keep_it_all_in_memory()
        WholeDataset.keep_it_all_in_memory(h5_file)

    npy_path = settings.large_file_folder + "datasets/OurAerial_preloadedImgs_BALCLASS.npy"

    I_WANT_TO_RECOMPUTE_THE_LABELS = False
    if I_WANT_TO_RECOMPUTE_THE_LABELS:
        assert False # don't want to mistakenly recompute these ...
        WholeDataset.compute_per_tile_class_in_batches()
        WholeDataset.save_per_tile_class(npy_path)

    WholeDataset.load_per_tile_class(npy_path)

    WholeDataset.report()

    return WholeDataset



def get_unbalanced_dataset(in_memory=False):
    assert in_memory == False

    # prep to move the dataset to >> /cluster/work/igp_psr/ruzickav <<
    # instead of loading indiv files, load batches in h5 files

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

    datasetInstance = DatasetInstance_OurAerial.DatasetInstance_OurAerial(settings, dataLoader, "256_cleanManual")

    # ! this one loads them all (CHECK: would some be deleted?)
    paths = datasetInstance.load_dataset_ONLY_PATHS_UPDATE_FROM_THE_OTHER_ONE_IF_NEEDED()
    lefts_paths, rights_paths, labels_paths = paths
    print("Paths: L,R,Y ", len(lefts_paths), len(rights_paths), len(labels_paths))

    WholeDataset.initialize_from_just_paths(paths)

    if in_memory:
        WholeDataset.keep_it_all_in_memory()

    npy_path = settings.large_file_folder + "datasets/OurAerial_preloadedImgs_unBALCLASS.npy"

    I_WANT_TO_RECOMPUTE_THE_LABELS = False
    if I_WANT_TO_RECOMPUTE_THE_LABELS:
        assert False # don't want to mistakenly recompute these ...
        WholeDataset.compute_per_tile_class_in_batches()
        WholeDataset.save_per_tile_class(npy_path)

    WholeDataset.load_per_tile_class(npy_path)

    WholeDataset.report()

    return WholeDataset