k = 5
SOURCE_L = "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2012_strip"+str(k)+"_256x256_over32_png/"
SOURCE_R = "/home/pf/pfstaff/projects/ruzicka/TiledDataset_256x256_32ov/2015_strip"+str(k)+"_256x256_over32_png/"
SOURCE_Y = "/home/pf/pfstaff/projects/ruzicka/CleanedVectors_manually_256x256_32over/vector_strip"+str(k)+"_256x256_over32/"

import numpy as np

import Settings
import mock
args = mock.Mock()
args.name = "test"
settings = Settings.Settings(args)
import DataLoader, DataPreprocesser, Debugger
import DatasetInstance_OurAerial
dataLoader = DataLoader.DataLoader(settings)
datasetInstance = DatasetInstance_OurAerial.DatasetInstance_OurAerial(settings, dataLoader, "256_cleanManual")

#"""
paths_2012 = [SOURCE_L]
paths_2015 = [SOURCE_R]
paths_vectors = [SOURCE_Y]

files_paths_2012 = datasetInstance.load_path_lists(paths_2012)
all_2012_png_paths, edge_tile_2012, total_tiles_2012 = datasetInstance.process_path_lists(files_paths_2012, paths_2012)
files_paths_2015 = datasetInstance.load_path_lists(paths_2015)
all_2015_png_paths, _, _ = datasetInstance.process_path_lists(files_paths_2015, paths_2015)

files_vectors = datasetInstance.load_path_lists(paths_vectors)
all_vector_paths = datasetInstance.process_path_lists_for_vectors(files_vectors, paths_vectors, edge_tile_2012, total_tiles_2012)



print("len(all_2012_png_paths):", len(all_2012_png_paths))
print("len(all_2015_png_paths):", len(all_2015_png_paths))
print("len(all_vector_paths):", len(all_vector_paths))


tmp_saved_data_for_inspection = all_2012_png_paths, all_2015_png_paths, all_vector_paths
tmp_saved_data_for_inspection = np.asarray(tmp_saved_data_for_inspection)

np.save("tmp_saved_data_for_inspection.npy", tmp_saved_data_for_inspection)
#"""

tmp_saved_data_for_inspection = np.load("tmp_saved_data_for_inspection.npy")

all_2012_png_paths, all_2015_png_paths, all_vector_paths = tmp_saved_data_for_inspection

print("len(all_2012_png_paths):", len(all_2012_png_paths))
print("len(all_2015_png_paths):", len(all_2015_png_paths))
print("len(all_vector_paths):", len(all_vector_paths))

# k 4 -> 2292 .. 2313
#     -> 2662 .. 2684 /
#     -> 3032 .. 3054 /
#     -> 3403 .. 3426 /
#     -> 3783 .. 3793 /
#     -> 4517 .. 4530 /
#     -> 4886 .. 4900 /
#     -> 5256 .. 5268 /
#     -> 5626 .. 5637 /
#     -> 5996 .. 6007 /
#     -> 6366 .. 6378 /
#     -> 6735 .. 6747 /
#     -> 7105 .. 7118 /
#     -> 7475 .. 7487 /
#     -> 7846 .. 7856 /
#     -> 8216 .. 8225 /
#     -> 8586 .. 8592 /
#     -> 8957 .. 8963 /

# 395 minus some amount on sides
# k 5 -> 75 .. 83 /
#     -> 467 .. 478 /
#     -> 859 .. 871 /
#     -> 1251 .. 1262 /
#     -> 1642 .. 1654 /
#     -> 2034 .. 2048 /
#     -> 2428 .. 2439 /
#     -> 2819 .. 2830 /

# also errors:
# strip4-2012_5740.PNG
# strip4-2012_7331.PNG
# strip4-2012_8333.PNG
# 7983 strip4-2012_8817.PNG (const site)
#file = open("exclude.txt", "a")

for i in range(0 , 10 +1):
    #if all_vector_paths[i]: # None are the no label ones
    #    t = all_2012_png_paths[i].split("/")[-1]+"\n"
    #    print(t)
    #    file.write(t)
    #continue

    if all_vector_paths[i]: # None are the no label ones
        print(i)
        print(all_2012_png_paths[i])
        print(all_2015_png_paths[i])
        print(all_vector_paths[i])

        txt = all_2012_png_paths[i].split("/")[-1]

        datasetInstance.debugger.viewTrippleFromUrl(all_2012_png_paths[i], all_2015_png_paths[i], all_vector_paths[i], optional_title=txt)

        print("--------------/n")

#file.close()


"""
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
"""