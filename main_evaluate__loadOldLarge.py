import os
import numpy as np
import h5py
import sklearn.metrics
import matplotlib.pyplot as plt




# File helpers

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_arr(arr, specialname = ""):

    suceeded = False

    while not suceeded:
        try:

            mkdir("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/"+"debuggerstuffs")
            hdf5_path = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/"+"debuggerstuffs/savedarr"+specialname+".h5"

            hdf5_file = h5py.File(hdf5_path, mode='w')
            hdf5_file.create_dataset("arr", data=arr, dtype="float32")
            hdf5_file.close()

            #print("Saved arr to:", hdf5_path)

            suceeded = True

        except Exception as e:
            print("exception, retrying e=",e)

            suceeded = False

def load_arr(specialname = ""):
    suceeded = False

    while not suceeded:
        try:

            hdf5_path = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/"+"debuggerstuffs/savedarr"+specialname+".h5"

            hdf5_file = h5py.File(hdf5_path, "r")
            arr = hdf5_file['arr'][:]
            hdf5_file.close()

            suceeded = True

        except Exception as e:
            print("exception, retrying e=",e)

            suceeded = False

    return arr


def mask_label_into_class_label(mask_labels, img_resolution=256, bigger_than_percent=3.0):
    array_of_number_of_change_pixels = []

    for mask in mask_labels:
        number_of_ones = np.count_nonzero(mask.flatten())  # << loading takes care of this 0 vs non-zero
        array_of_number_of_change_pixels.append(number_of_ones)

    save_arr(array_of_number_of_change_pixels, "BALANCING")
    array_of_number_of_change_pixels = load_arr("BALANCING")

    array_of_number_of_change_pixels = array_of_number_of_change_pixels / (
            img_resolution * img_resolution) * 100.0  # percentage of image changed

    class_labels = []
    for value in array_of_number_of_change_pixels:
        is_change = value > bigger_than_percent
        class_labels.append(int(is_change))

    return np.array(class_labels)

def human_legible_tiles_report(predicted_orig, labels_orig, wanted_recall, thresholds):
        
    labels = np.array(labels_orig, copy=True)
    test_Tiles = mask_label_into_class_label(labels)
    arr_gts = test_Tiles.flatten()

    N = len(arr_gts)
    # worst case scenario:
    best_recall_cost = N
    best_recall_idx = 0
	
    recalls = []

    for i, thr in reversed(list(enumerate(thresholds))):
        r = 0
        predictions_thresholded = np.array(predicted_orig, copy=True)
        for image in predictions_thresholded:
            image[image >= thr] = 1
            image[image < thr] = 0
        predicted_Tiles = mask_label_into_class_label(predictions_thresholded)
        arr_predictions = predicted_Tiles.flatten()
        r = sklearn.metrics.recall_score(arr_gts, arr_predictions)
        recalls.append(r)

        if r > wanted_recall:
            # cost = how many tiles we have to check = TP+FP

            conf = sklearn.metrics.confusion_matrix(arr_gts, arr_predictions)
            TP = conf[1][1]
            TN = conf[0][0]
            FP = conf[0][1]
            FN = conf[1][0]
            #N = TP + TN + FP + FN

            cost_r = (TP + FP)

            if cost_r <= best_recall_cost:
                best_recall_cost = cost_r
                best_recall_idx = i

    cost_perc = 100*(best_recall_cost/N)
    report_str = "If we want the recall to be better than "+str(wanted_recall)+\
                     ", we need to set the threshold to be = "+str(thresholds[best_recall_idx])+" which will give us " \
                     "recall of "+str(recalls[best_recall_idx])+" while the number of tiles needed to check is "+\
                     str(best_recall_cost)+" from the worst case scenario "+str(N)+" (that's "+str(np.round(cost_perc, 2))+"%).\n\n"

    print(report_str)
    return report_str, cost_perc

def human_legible_as_a_plot(predicted_orig, labels_orig,  thresholds, plot_filename=""):
    # Plot x=wanted_recall, y=cost (as % of the orig dataset needed to check)

    ys = []
    wanted_txt = ""
    for thr in thresholds:
        wanted_recall = thr

        txt, cost_perc = human_legible_tiles_report(predicted_orig, labels_orig, wanted_recall, thresholds)
        wanted_txt += txt

        ys.append(cost_perc)

    xs = thresholds

    plt.figure() # figsize=(w, h)

    print("xs", len(xs), xs)
    print("ys", len(ys), ys)
    lw = 2

    plt.title('Cost for given wanted recall')
    plt.xlabel('wanted recall')
    plt.ylabel('cost (in percents of the original dataset)')

    plt.plot(xs, ys, color='red', marker='o', lw=lw, label="Cost")
    plt.legend()

    plt.ylim(0.0, 100.0) # in percent

    plt.savefig(plot_filename+'_Costs.png')
    plt.close()

    return wanted_txt, xs, ys

	
model_idx = 0 # 0 to 5?
for model_idx in [2, 3, 4]: # 0 and 1 are done
    
    path_large_files_backup_sol = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/main_eval_mem_issues/"

    folder_name = "weightsModel2_cleanManual_100ep_ImagenetWgenetW_resnet50-16batch_Augmentation1to1_ClassWeights1to3_TestVal_[KFold_"+str(model_idx)+"z5]"

    if not os.path.exists(path_large_files_backup_sol):
       os.makedirs(path_large_files_backup_sol)
    if not os.path.exists(path_large_files_backup_sol + folder_name + "/"):
       os.makedirs(path_large_files_backup_sol + folder_name + "/")
    predicted_total = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_predicted_total.npy")
    gts_total = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_gts_total.npy")
    statistics = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_statistics_total.npy")


    mask_stats, tiles_stats = statistics
    tiles_best_thr, tiles_selected_recall, tiles_selected_precision, tiles_selected_accuracy, tiles_selected_f1 = tiles_stats
    pixels_best_thr, pixels_selected_recall, pixels_selected_precision, pixels_selected_accuracy, pixels_selected_f1, pixels_auc = mask_stats


    print("Thresholds were:")
    print("tiles_best_thr=",tiles_best_thr)
    print("pixels_best_thr=",pixels_best_thr)


    print("predicted_total.shape=",predicted_total.shape)
    print("gts_total.shape=",gts_total.shape)

	
    # PER TILE! ====================
        
    threshold = tiles_best_thr
	
    ground_truths_classlabels = mask_label_into_class_label(gts_total)
    del gts_total
    predicted_total_thresholded = np.array(predicted_total)
    del predicted_total
    for image in predicted_total_thresholded:
        image[image >= threshold] = 1
        image[image < threshold] = 0
    predicted_total_classlabels = mask_label_into_class_label(predicted_total_thresholded)

    del predicted_total_thresholded

    print("in the middle we have:", len(predicted_total_classlabels), len(ground_truths_classlabels))
	
    tiles_accuracy = sklearn.metrics.accuracy_score(ground_truths_classlabels, predicted_total_classlabels)
    print("tiles_accuracy=", tiles_accuracy)
    tiles_precision = sklearn.metrics.precision_score(ground_truths_classlabels, predicted_total_classlabels)
    print("tiles_precision=", tiles_precision)
    tiles_recall = sklearn.metrics.recall_score(ground_truths_classlabels, predicted_total_classlabels)
    print("tiles_recall=", tiles_recall)
    tiles_f1 = sklearn.metrics.f1_score(ground_truths_classlabels, predicted_total_classlabels)
    print("tiles_f1=", tiles_f1)

	# BONUS STATS ON TILES === conf, TPR and FPR
    labels = ["no change", "change"]
    report = str(sklearn.metrics.classification_report(ground_truths_classlabels, predicted_total_classlabels, target_names=labels))
    print(report)
    conf = sklearn.metrics.confusion_matrix(ground_truths_classlabels, predicted_total_classlabels)
    #     Thus in binary classification, the count of true negatives is
    #     :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    #     :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
    conf_str = str(conf)
    conf_str += str("\nas [[TN   FP], [FN   TP]]\nTP "+str(conf[1][1])+" \t ... correctly classified as a change.\n" \
                "TN "+str(conf[0][0])+"\t ... correctly classified as a no-change.\n" \
                "FP "+str(conf[0][1])+"\t ... classified as change while it's not.\n" \
                "FN "+str(conf[1][0])+"\t ... classified as no-change while it is one.")

    TP = conf[1][1]
    TN = conf[0][0]
    FP = conf[0][1]
    FN = conf[1][0]

    # TPR (True Positive Rate) = # True positives / # positives = Recall = TP / (TP+FN)
    # FPR (False Positive Rate) = # False Positives / # negatives = FP / (FP+TN)
    tiles_TruePositiveRate = TP / (TP+FN)
    tiles_FalsePositiveRate = FP / (FP+TN)

    conf_str += "TruePositiveRate = TP / (TP+FN) = "+str(tiles_TruePositiveRate)+"\n"
    conf_str += "FalsePositiveRate = FP / (FP+TN) = "+str(tiles_FalsePositiveRate)+"\n"

    print(conf_str)
	
	# BONUS STATS ON TILES === annotation cost
    predicted_total = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_predicted_total.npy")
    gts_total = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_gts_total.npy")

    #threshold_fineness = 0.05
    #thresholds = np.arange(0.0, 1.0+threshold_fineness, threshold_fineness)
    threshold_fineness = 0.01
    thresholds = np.arange(0.75, 1.0+threshold_fineness, threshold_fineness)
    _, AnnotCosts_xs, AnnotCosts_ys = human_legible_as_a_plot(predicted_total, gts_total, thresholds, plot_filename=path_large_files_backup_sol + folder_name + "/")
	
    # PER PIXEL! ====================
    pixels_auc = 0
    pixels_accuracy = 0
    pixels_precision = 0
    pixels_recall = 0
    pixels_f1 = 0

    # """
    predicted_total = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_predicted_total.npy")
    gts_total = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_gts_total.npy")

	
	
    # Independently AUC
    print("calculating flattens...")
    #unthresholded_flat = predicted_total.flatten()
    #gts_flat = gts_total.flatten()
    unthresholded_flat = predicted_total.ravel()
    gts_flat = gts_total.ravel()

    del predicted_total
    del gts_total
    print("calculating auc...")
    
    pixels_auc = sklearn.metrics.roc_auc_score(gts_flat, unthresholded_flat)
    print("pixels_auc=", pixels_auc)

    del unthresholded_flat
    del gts_flat

    predicted_total = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_predicted_total.npy")
    gts_total = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_gts_total.npy")
    
    #pixels_auc = 0
    # Then the rest of the stats (recall, acc, prec, f1)
    
    threshold = pixels_best_thr
	
    #del gts_total
    predicted_total_thresholded = np.array(predicted_total)
    del predicted_total
    for image in predicted_total_thresholded:
        image[image >= threshold] = 1
        image[image < threshold] = 0

    #ground_truths_flat = gts_total.flatten()
    #predicted_flat = predicted_total_thresholded.flatten()
    ground_truths_flat = gts_total.ravel()
    predicted_flat = predicted_total_thresholded.ravel()
    del gts_total
    del predicted_total_thresholded

    print("in the middle we have:", len(predicted_flat), len(ground_truths_flat))
	
    pixels_accuracy = sklearn.metrics.accuracy_score(ground_truths_flat, predicted_flat)
    print("pixels_accuracy=", pixels_accuracy)
    pixels_precision = sklearn.metrics.precision_score(ground_truths_flat, predicted_flat)
    print("pixels_precision=", pixels_precision)
    pixels_recall = sklearn.metrics.recall_score(ground_truths_flat, predicted_flat)
    print("pixels_recall=", pixels_recall)
    pixels_f1 = sklearn.metrics.f1_score(ground_truths_flat, predicted_flat)
    print("pixels_f1=", pixels_f1)
    # """
    

    statistics_pixels = pixels_recall, pixels_precision, pixels_accuracy, pixels_f1, pixels_auc
    
    statistics_tiles = tiles_recall, tiles_precision, tiles_accuracy, tiles_f1, tiles_TruePositiveRate, tiles_FalsePositiveRate, AnnotCosts_xs, AnnotCosts_ys
    
    statistics_we_care_about = statistics_pixels, statistics_tiles
    
    statistics_we_care_about = np.asarray(statistics_we_care_about)

    np.save(path_large_files_backup_sol + folder_name + "/"+"calculated_pixel_statistics_1PERCPER.npy", statistics_we_care_about)
    
    file = open(path_large_files_backup_sol + folder_name + "/"+"report_1PERCPER.txt", "w")
    file.write(report+"\n")
    file.write(conf_str+"\n")
    file.close()

