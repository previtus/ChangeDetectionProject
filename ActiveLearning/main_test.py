from ActiveLearning.LargeDatasetHandler_AL import LargeDatasetHandler_AL
import Settings
from timeit import default_timer as timer


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

# ! this one automatically balances the data + deletes misfits in the resolution
#data, paths = datasetInstance.load_dataset()
#lefts_paths, rights_paths, labels_paths = paths
#print("Paths: L,R,Y ", len(lefts_paths), len(rights_paths), len(labels_paths))


# ! this one loads them all (CHECK: would some be deleted?)
paths = datasetInstance.load_dataset_ONLY_PATHS_UPDATE_FROM_THE_OTHER_ONE_IF_NEEDED()
lefts_paths, rights_paths, labels_paths = paths
print("Paths: L,R,Y ", len(lefts_paths), len(rights_paths), len(labels_paths))

WholeDataset.initialize_from_just_paths(paths)
WholeDataset.report()

#WholeDataset.keep_it_all_in_memory()
#WholeDataset.keep_it_all_in_memory(h5_file)

WholeDataset.report()

nb_epoch = 50 # stability testing now ...
for e in range(nb_epoch):
    print("epoch %d" % e)
    start = timer()

    # mode > indices, dataonly, datalabels
    for batch in WholeDataset.generator_for_all_images(2048, mode='datalabels'): # Yields a large batch sample
        indices = batch[0]
        #model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)
        #print("batch this ", len(indices), "into 32 in Keras")
        print("indices from", indices[0], "to", indices[-1])

        datas = batch[1] # lefts and rights
        lefts = datas[0]
        #print("left data from", lefts[0].shape, "to", lefts[-1].shape)

        labels = batch[2] # labels
        #print("labels from", labels[0].shape, "to", labels[len(labels)-1].shape)

    end = timer()
    time = (end - start)
    print("This epoch took "+str(time)+"s ("+str(time/60.0)+"min) [ << Right now just loading images ]")

    # This epoch took 1856.6551471220446s (30.944252452034075min) [ << Right now just loading images ]
    # This epoch took 1745.6752279539942s (29.09458713256657min) [ << Right now just loading images ]
    # ... etc a lot ...
    # This epoch took 1497.658740325016s (24.960979005416934min) [ << Right now just loading images ]
    # This epoch took 2146.4903272199444s (35.77483878699908min) [ << Right now just loading images ]

    #break
