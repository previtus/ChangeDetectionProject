# === Initialize sets - Unlabeled, Train and Test
import keras
from ActiveLearning.LargeDatasetHandler_AL import get_balanced_dataset, get_unbalanced_dataset
from timeit import default_timer as timer
from datetime import *
import os

months = ["unk", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

def main():

    import matplotlib.pyplot as plt
    import numpy as np

    RemainingUnlabeledSet = get_unbalanced_dataset()

    """
    PER_BATCH = 2048 # How big files would we like ? test with smaller and if we can move that comfortably to a server ...
    for batch in RemainingUnlabeledSet.generator_for_all_images(PER_BATCH,mode='SAVEBATCHFILES', skip_i_batches=40):
        print(">>>>>>>>>>>>>>>>>>>>>>>>> processed batch " + batch)

        #break
    """

    k = 0

    PER_BATCH = 2048  # How big files would we like ? test with smaller and if we can move that comfortably to a server ...
    for batch in RemainingUnlabeledSet.generator_for_all_images(PER_BATCH, mode='dataonly_LOADBATCHFILES'):
        # batch < corresponding_indices, [lefts,rights]
        remaining_indices = batch[0]
        remaining_data = batch[1]  # lefts and rights, no labels!
        print("MegaBatch (", RemainingUnlabeledSet.N_of_data,") of size", len(remaining_indices), " = indices from id", remaining_indices[0], "to id", remaining_indices[-1])
        L, R = remaining_data
        L = np.asarray(L).astype('float32')
        R = np.asarray(R).astype('float32')

        print("L", L.shape, "R", R.shape)

        del L
        del R
        del remaining_data
        del remaining_indices
        del batch

        #k += 1
        #if k > 10:
        #    break

if __name__ == '__main__':
    start = timer()

    main()

    end = timer()
    time = (end - start)

    print("This run took " + str(time) + "s (" + str(time / 60.0) + "min)")

    import keras

    keras.backend.clear_session()
