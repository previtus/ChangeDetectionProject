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

statistics_over_models_incUnbal = []

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
    statistics_over_models_incUnbal.append(statistics_we_care_about)
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



####################################### Save as main_evaluate plot:
add_text = "plotOldLarge_fromResNet50OnUnbalanced"
model_backend = "ResNet50"

print("Overall statistics::: (", len(statistics_over_models_incUnbal), ")")
print(statistics_over_models_incUnbal)

# each model has [mask_stats, tiles_stats] = [[thr, recall, precision, accuracy, f1], [...]]
tiles_recalls = []
tiles_precisions = []
tiles_accuracies = []
tiles_f1s = []

mask_recalls = []
mask_precisions = []
mask_accuracies = []
mask_f1s = []
mask_AUCs = []

#         tiles_stats = tiles_best_thr, tiles_selected_recall, tiles_selected_precision, tiles_selected_accuracy, tiles_selected_f1
#         mask_stats = pixels_best_thr, pixels_selected_recall, pixels_selected_precision, pixels_selected_accuracy, pixels_selected_f1, pixels_auc
#         statistics = mask_stats, tiles_stats

for stats in statistics_over_models_incUnbal:
    mask_stats, tiles_stats = stats

    tiles_recalls.append(tiles_stats[0])
    mask_recalls.append(mask_stats[0])

    tiles_precisions.append(tiles_stats[1])
    mask_precisions.append(mask_stats[1])

    tiles_accuracies.append(tiles_stats[2])
    mask_accuracies.append(mask_stats[2])

    tiles_f1s.append(tiles_stats[3])
    mask_f1s.append(mask_stats[3])

    mask_AUCs.append(mask_stats[4])

# REPORT
report_text = ""
report_text += "Tiles evaluation:\n"
report_text += "mean tiles_recalls = " + str(100.0 * np.mean(tiles_recalls)) + " +- " + str(
    100.0 * np.std(tiles_recalls)) + " std \n"
report_text += "mean tiles_precisions = " + str(100.0 * np.mean(tiles_precisions)) + " +- " + str(
    100.0 * np.std(tiles_precisions)) + " std \n"
report_text += "mean tiles_accuracies = " + str(100.0 * np.mean(tiles_accuracies)) + " +- " + str(
    100.0 * np.std(tiles_accuracies)) + " std \n"
report_text += "mean tiles_f1s = " + str(100.0 * np.mean(tiles_f1s)) + " +- " + str(
    100.0 * np.std(tiles_f1s)) + " std \n"
report_text += "\n"
report_text += "Mask evaluation:\n"
report_text += "mean mask_recalls = " + str(100.0 * np.mean(mask_recalls)) + " +- " + str(
    100.0 * np.std(mask_recalls)) + " std \n"
report_text += "mean mask_precisions = " + str(100.0 * np.mean(mask_precisions)) + " +- " + str(
    100.0 * np.std(mask_precisions)) + " std \n"
report_text += "mean mask_accuracies = " + str(100.0 * np.mean(mask_accuracies)) + " +- " + str(
    100.0 * np.std(mask_accuracies)) + " std \n"
report_text += "mean mask_f1s = " + str(100.0 * np.mean(mask_f1s)) + " +- " + str(100.0 * np.std(mask_f1s)) + " std \n"
report_text += "mean mask_AUCs = " + str(100.0 * np.mean(mask_AUCs)) + " +- " + str(
    100.0 * np.std(mask_AUCs)) + " std \n"

file = open("evaluation_plots/report_boxplotStats_" + add_text + ".txt", "w")
file.write(report_text)
file.close()

xs = ["recall", "precision", "accuracy", "f1"]
data = [tiles_recalls, tiles_precisions, tiles_accuracies, tiles_f1s]

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
"""
ax = sns.boxplot(x=xs, y=data)
ax.set_title('Stats per tiles')
ax.set_ylim(0.0,1.0)
plt.show()
"""

fig1, ax1 = plt.subplots()
ax1.set_title('KFoldCrossval statistics (per tiles) - ' + model_backend)
ax1.boxplot(data, labels=xs)
ax1.set_ylim(0.0, 1.0)
# plt.show()
plt.savefig("evaluation_plots/boxplot_tiles_stats_" + add_text + ".png")
plt.savefig("evaluation_plots/boxplot_tiles_stats_" + add_text + ".pdf")

fig2, ax2 = plt.subplots()
ax2.set_title('KFoldCrossval statistics (per pixels) - ' + model_backend)
xs_pixels = ["recall", "precision", "accuracy", "f1", "AUC"]
data = [mask_recalls, mask_precisions, mask_accuracies, mask_f1s, mask_AUCs]
ax2.boxplot(data, labels=xs_pixels)
ax2.set_ylim(0.0, 1.0)
# plt.show()
plt.savefig("evaluation_plots/boxplot_masks_stats_" + add_text + ".png")
plt.savefig("evaluation_plots/boxplot_masks_stats_" + add_text + ".pdf")

