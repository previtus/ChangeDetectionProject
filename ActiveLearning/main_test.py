from ActiveLearning.LargeDatasetHandler_AL import LargeDatasetHandler_AL
import Settings

# init structures
import mock
args = mock.Mock()
args.name = "test"

settings = Settings.Settings(args)
WholeDataset = LargeDatasetHandler_AL(settings)
WholeDataset.report()

# load paths of our favourite dataset!

import DataLoader, DataPreprocesser, Debugger
import DatasetInstance_OurAerial

dataLoader = DataLoader.DataLoader(settings)
debugger = Debugger.Debugger(settings)

h5_file = settings.large_file_folder + "datasets/OurAerial_preloadedImgs_subBAL3.0_1.0_sel2144_res256x256.h5"

datasetInstance = DatasetInstance_OurAerial.DatasetInstance_OurAerial(settings, dataLoader, "256_cleanManual")
dataPreprocesser = DataPreprocesser.DataPreprocesser(settings, datasetInstance)

# ! this one automatically balances the data + deletes misfits in the resolution
data, paths = datasetInstance.load_dataset()
lefts_paths, rights_paths, labels_paths = paths
print("Paths: L,R,Y ", len(lefts_paths), len(rights_paths), len(labels_paths))


# ! this one loads them all (CHECK: would some be deleted?)
#paths = datasetInstance.load_dataset_ONLY_PATHS_UPDATE_FROM_THE_OTHER_ONE_IF_NEEDED()
#lefts_paths, rights_paths, labels_paths = paths
#print("Paths: L,R,Y ", len(lefts_paths), len(rights_paths), len(labels_paths))

WholeDataset.initialize_from_just_paths(paths)
WholeDataset.report()

#WholeDataset.keep_it_all_in_memory()
#WholeDataset.keep_it_all_in_memory(h5_file)

WholeDataset.report()

nb_epoch = 50
for e in range(nb_epoch):
    print("epoch %d" % e)
    # mode > indices, dataonly, datalabels
    for batch in WholeDataset.generator_for_all_images(1000, mode='datalabels'): # Yields a large batch sample
        indices = batch[0]
        #model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)
        print("batch this ", len(indices), "into 32 in Keras")
        print("indices from", indices[0], "to", indices[-1])

        datas = batch[1] # lefts and rights
        lefts = datas[0]
        print("left data from", lefts[0].shape, "to", lefts[-1].shape)

        labels = batch[2] # labels
        print("labels from", labels[0].shape, "to", labels[len(labels)-1].shape)



    break
