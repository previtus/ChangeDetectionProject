from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

import numpy as np
from timeit import default_timer as timer
import cv2
from skimage.transform import resize
import os

#pool = Pool(os.cpu_count() - 1)
#pool = ThreadPool(4)


def BALD_diff(pixel_predictions):
    # Bayesian Active Learning by Disagreement = BALD = https://arxiv.org/abs/1112.5745
    # T = len(pixel_predictions) < is the shape of the number of predictions
    # assert len(pixel_predictions.shape) == 1

    #def baldlogs(val):
    #    return - val * np.log(val) - (1 - val) * np.log(1 - val)

    #accums = np.apply_along_axis(arr=pixel_predictions, axis=0, func1d=baldlogs) # 13s
    #accum = np.sum(accums)

    accum = 0
    #accums = []
    for val in pixel_predictions:
        accum += - val * np.log(val) - (1 - val) * np.log(1 - val) # 6-8s
        #accums.append( - val * np.log(val) - (1 - val) * np.log(1 - val) ) # 9s
    #accum = np.sum(accums)

    return accum


def ent_img_sumDiv(pixel_predictions):
    return np.sum(pixel_predictions, axis=0) / len(pixel_predictions)


def ent_img_log(pk):
    return - pk * np.log(pk)


def multithr_calc_metrics(predictions_one_samples, acquisition_function = "BALD"):
    sum_bald = 0
    sum_ent = 0

    if acquisition_function == "Entropy" or acquisition_function == "BALD":
        ent_img_pk0 = np.apply_along_axis(arr=predictions_one_samples, axis=0, func1d=ent_img_sumDiv)
        ent_img_pk1 = np.ones_like(ent_img_pk0) - ent_img_pk0
        ent_img_ent0 = np.apply_along_axis(arr=ent_img_pk0, axis=0, func1d=ent_img_log)
        ent_img_ent1 = np.apply_along_axis(arr=ent_img_pk1, axis=0, func1d=ent_img_log)
        entropy_image = ent_img_ent0 + ent_img_ent1
        sum_ent = np.sum(entropy_image.flatten())

    if acquisition_function == "BALD":
        bald_diff_image = np.apply_along_axis(arr=predictions_one_samples, axis=0, func1d=BALD_diff)
        bald_image = -1 * (entropy_image - bald_diff_image)
        sum_bald = np.sum(bald_image.flatten())

    variance_image = np.var(predictions_one_samples, axis=0)
    sum_var = np.sum(variance_image.flatten())

    return sum_bald, sum_ent, sum_var


def function_to_apply(item):
    res = resize(item[0], (140, 54), anti_aliasing=True, mode='constant')
    return res


some_image = "/home/ruzickav/somearetrickyiteration_01_randomlyselected_28.png"
#some_image = "[CompRuns1_Seed50_MCBN10_Var_S]_dbg_last_al_balance_plot.png"

im = cv2.imread(some_image)[:,:,0]
im = resize(im, (256,256), anti_aliasing=True, mode='constant')

im = im / 500.0
im = im + np.ones_like(im)*0.5
print("im", im.shape)

some_image = "/home/ruzickav/oscd_withSigmoidNow.png"
#some_image = "[CompRuns1_Seed50_MCBN10_Var_S]_dbg_last_al_big_plot_tilesScores.png"

im2 = cv2.imread(some_image)[:,:,0]
im2 = resize(im, (256,256), anti_aliasing=True, mode='constant')

im2 = im2 / 300.0
im2 = im2 + np.ones_like(im2)*0.6
print("im2", im2.shape)


f = im.flatten()
print(min(f),max(f))

f = im2.flatten()
print(min(f),max(f))


Ntwo = 10
array_to_be_processed = [[im, im2, im, im2, im]] * Ntwo + [[im, im, im, im, im]] * Ntwo
array_to_be_processed = np.asarray(array_to_be_processed)

print(array_to_be_processed.shape)

# Multi processing ============================================================================================
start = timer()

#array_of_results = pool.map(function_to_apply, array_to_be_processed)

with Pool() as pool:
    array_of_results = pool.map(multithr_calc_metrics, array_to_be_processed)


#array_of_results = pool.map(lambda i: (
#    multithr_calc_metrics(i)
#), array_to_be_processed)


end = timer()
print("Multiprocessing Time", (end-start))


# Single processing ============================================================================================
start = timer()

array_of_results_single = []
for i in array_to_be_processed:
    array_of_results_single.append(multithr_calc_metrics(i))

end = timer()
print("Singleprocessing Time", (end-start))

## For example 4 times speedup

array_of_results_single = np.asarray(array_of_results_single)
array_of_results = np.asarray(array_of_results)

print("Equal?:", np.array_equal(array_of_results, array_of_results_single) )
#print(array_of_results)
#print(array_of_results_single)

print(array_of_results.shape)
print(array_of_results_single.shape)

print(array_of_results[0])
print(array_of_results[-1])
#print(array_of_results_single)