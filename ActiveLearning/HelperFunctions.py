import h5py
import numpy as np
from skimage import io

def load_images_from_h5(hdf5_path):
    hdf5_file = h5py.File(hdf5_path, "r")
    lefts = hdf5_file['lefts'][:]
    rights = hdf5_file['rights'][:]
    labels = hdf5_file['labels'][:]
    hdf5_file.close()

    return lefts, rights, labels

def load_vector_image(filename, IMAGE_RESOLUTION=256):
    if filename == None:
        arr = np.zeros((IMAGE_RESOLUTION, IMAGE_RESOLUTION), dtype=float)
        return arr

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
    img = io.imread(filename)
    arr = np.asarray(img)
    return arr