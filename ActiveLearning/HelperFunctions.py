import h5py
import numpy as np
from skimage import io

SERVER_HAX = False

def save_images_to_h5_DEFAULT_DATA_FORMAT(lefts, rights, labels, hdf5_path):
    SUBSET = len(lefts)

    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset("lefts", data=lefts)
    hdf5_file.create_dataset("rights", data=rights)
    hdf5_file.create_dataset("labels", data=labels)
    hdf5_file.close()

    print("Saved", SUBSET, "images successfully to:", hdf5_path)

    return hdf5_path
def load_images_from_h5(hdf5_path):
    hdf5_file = h5py.File(hdf5_path, "r")
    lefts = hdf5_file['lefts'][:]
    rights = hdf5_file['rights'][:]
    labels = hdf5_file['labels'][:]
    hdf5_file.close()

    return lefts, rights, labels


#GLOBAL_COMPARATOR = []

def save_images_to_h5_with_wholeDataset_indices(lefts, rights, labels, indices, hdf5_path):
    SIZE = len(lefts)

    #global GLOBAL_COMPARATOR
    #GLOBAL_COMPARATOR = indices

    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset("lefts", data=lefts)
    hdf5_file.create_dataset("rights", data=rights)
    hdf5_file.create_dataset("labels", data=labels)
    hdf5_file.create_dataset("indices", data=indices, dtype="int") # up to 81000
    hdf5_file.close()

    print("Saved", SIZE, "images successfully to:", hdf5_path)

    return hdf5_path

def load_images_from_h5_with_wholeDataset_indices(hdf5_path):
    hdf5_file = h5py.File(hdf5_path, "r")
    lefts = hdf5_file['lefts'][:]
    rights = hdf5_file['rights'][:]
    labels = hdf5_file['labels'][:]
    indices = hdf5_file['indices'][:]
    hdf5_file.close()

    #global GLOBAL_COMPARATOR
    #equal = np.array_equal(indices, GLOBAL_COMPARATOR)
    #print("equal", equal)
    #iasjdasl

    return lefts, rights, labels, indices


def load_vector_image(filename, IMAGE_RESOLUTION=256):
    if filename == None:
        arr = np.zeros((IMAGE_RESOLUTION, IMAGE_RESOLUTION), dtype=float)
        return arr

    if SERVER_HAX:
        if "/home/pf/pfstaff/projects/ruzicka/" in filename:
            ## /home/pf/pfstaff/projects/ruzicka/CleanedVectors_manually_256x256_32over/
            filename = "/cluster/work/igp_psr/ruzickav/ChangeDetectionProject_files/entire_dataset_files"+filename[33:] # starts with "/CleanedVectors_manually_256x256_32over...

    img = io.imread(filename)
    arr = np.asarray(img)

    ## FOR NEWER DATASETS
    arr[arr <= 0] = 0
    arr[arr == 65535] = 0  # hi ArcGis ghosts
    arr[arr != 0] = 1

    # anything <= 0 (usually just one value) => 0 (no change)
    # anything >  0                          => 1 (change)
    return arr


def load_raster_image(filename):
    if SERVER_HAX:
        if "/home/pf/pfstaff/projects/ruzicka/" in filename:
            filename = "/cluster/work/igp_psr/ruzickav/ChangeDetectionProject_files/entire_dataset_files"+filename[33:] # starts with "/TiledDataset_256x256_32ov...

    img = io.imread(filename)
    arr = np.asarray(img)
    return arr