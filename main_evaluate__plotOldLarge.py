import os
import numpy as np
import h5py
import sklearn.metrics
import matplotlib.pyplot as plt

# We really care only about:
# - pixel AUC
# - tile F1

collective_AUCs = []
collective_recalls = []

collective_yCOSTs = []
xs = []
collective_costs_selected = []

model_idx = 0 # 0 to 5?
for model_idx in [0, 1, 2, 3, 4]:
    
    path_large_files_backup_sol = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/main_eval_mem_issues/"

    folder_name = "weightsModel2_cleanManual_100ep_ImagenetWgenetW_resnet50-16batch_Augmentation1to1_ClassWeights1to3_TestVal_[KFold_"+str(model_idx)+"z5]"



    statistics_standard = np.load(path_large_files_backup_sol + folder_name + "/" + "BatchI-" + str(model_idx) + "_statistics_total.npy")
    mask_stats, tiles_stats = statistics_standard
    tiles_best_thr, tiles_selected_recall, tiles_selected_precision, tiles_selected_accuracy, tiles_selected_f1 = tiles_stats
    pixels_best_thr, pixels_selected_recall, pixels_selected_precision, pixels_selected_accuracy, pixels_selected_f1, pixels_auc = mask_stats
    print("Thresholds were:")
    print("tiles_best_thr=",tiles_best_thr)
    print("pixels_best_thr=",pixels_best_thr)


    
    statistics_we_care_about = np.load(path_large_files_backup_sol + folder_name + "/"+"calculated_pixel_statistics.npy")
    # FINER COST PLOTs:
    #statistics_we_care_about = np.load(path_large_files_backup_sol + folder_name + "/"+"calculated_pixel_statistics_1PERCPER.npy")

    statistics_pixels, statistics_tiles = statistics_we_care_about
    pixels_recall, pixels_precision, pixels_accuracy, pixels_f1, pixels_auc = statistics_pixels
    tiles_recall, tiles_precision, tiles_accuracy, tiles_f1, tiles_TruePositiveRate, tiles_FalsePositiveRate, AnnotCosts_xs, AnnotCosts_ys = statistics_tiles
    xs = AnnotCosts_xs
    
    collective_AUCs.append(pixels_auc)
    collective_recalls.append(tiles_recall)

    collective_yCOSTs.append(AnnotCosts_ys)

    print("xs", AnnotCosts_xs)
    print("ys", AnnotCosts_ys)
    for idx, thr_calc in enumerate(AnnotCosts_xs):
        if thr_calc == tiles_best_thr:
            cost = AnnotCosts_ys[idx]
            break
    collective_costs_selected.append(cost)

# print statistics over loaded processed files:
print("statistics over loaded processed files (up to ",model_idx,"):")

print("pixel AUC", np.mean(collective_AUCs), "+-", np.std(collective_AUCs))
print("tile recall", np.mean(collective_recalls), "+-", np.std(collective_recalls))
print("tile cost (percent of dataset)", np.mean(collective_costs_selected), "+-", np.std(collective_costs_selected))




plt.figure(figsize=(7, 7)) # w, h
print(xs)

COSTs_means = np.mean(collective_yCOSTs, 0)
COSTs_stds = np.std(collective_yCOSTs, 0)
print("collective_yCOSTs")
print("means:", COSTs_means)
print("stds:", COSTs_stds)
    
plt.plot(xs, COSTs_means, color="red", label="Cost")
plt.plot(xs, COSTs_means+COSTs_stds, color="gray")

tmp = COSTs_means-COSTs_stds
prev = 0
for i,val in enumerate(tmp):
    if val < prev:
        tmp[i] = prev
    prev = val
plt.plot(xs, tmp, color="gray")

plt.xticks(xs)

plt.xlabel('Required recall')
plt.ylabel('Percent of dataset needed to check')
plt.legend()
plt.ylim(0.0, 100)
plt.savefig("[FOOOO]_COSTs.png")
plt.savefig("[FOOOO]_COSTs.pdf")
plt.close()