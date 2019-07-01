import matplotlib, os
matplotlib.use("Agg")

from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt

results_folder = "/home/ruzickav/python_projects/ChangeDetectionProject/data/CompRuns1_data/"

ensemble_results = results_folder+"ensemble/"
mcbn_results = results_folder+"mcbn/"
random_results = results_folder+"random/"

ensemble_files = [ensemble_results+f for f in listdir(ensemble_results) if "npy" in f]
mcbn_files = [mcbn_results+f for f in listdir(mcbn_results) if "npy" in f]
random_files = [random_results+f for f in listdir(random_results) if "npy" in f]

plot_filepaths = [random_files, mcbn_files, ensemble_files]
plot_data = []
names = ["Random", "MCBN", "Ensemble"]
colors_means = ["red", "blue", "black"]
colors_stds = ["pink", "lightblue", "gray"]

for method_set_files in plot_filepaths:
    collective_Xs = []
    collective_AUCs = []
    collective_F1s = []
    collective_BalanceNoCh = []
    collective_BalanceYesCh = []
    for one_fold in method_set_files:
        statistics = np.load(one_fold)

        pixel_statistics, tile_statistics = statistics
        pixels_thresholds, xs_number_of_data, pixels_recalls, pixels_precisions, pixels_accuracies, pixels_f1s, Ns_changed, Ns_nochanged, pixels_AUCs = pixel_statistics
        tiles_thresholds, xs_number_of_data, tiles_recalls, tiles_precisions, tiles_accuracies, tiles_f1s, Ns_changed, Ns_nochanged = tile_statistics
    
        # keep what we care about together:
        # - AUC score from pixels (pixels_AUCs)
        # - (maybe? f1 from tiles) (tiles_f1s)
        # - balance ( Ns_changed, Ns_nochanged )
        collective_Xs.append(xs_number_of_data)
        collective_AUCs.append(pixels_AUCs)
        collective_F1s.append(tiles_f1s)
        collective_BalanceNoCh.append(Ns_nochanged)
        collective_BalanceYesCh.append(Ns_changed)
    
    print("\n\n")
    print(" --- ", len(method_set_files), " => ", method_set_files)
    print("collective_Xs",collective_Xs)
    print("collective_AUCs",collective_AUCs)
    print("collective_F1s",collective_F1s)
    print("collective_BalanceNoCh",collective_BalanceNoCh)
    print("collective_BalanceYesCh",collective_BalanceYesCh)

    smallest_x = min([len(a) for a in collective_Xs])
    collective_Xs = [a[:smallest_x] for a in collective_Xs]
    collective_AUCs = [a[:smallest_x] for a in collective_AUCs]
    collective_F1s = [a[:smallest_x] for a in collective_F1s]
    collective_BalanceNoCh = [a[:smallest_x] for a in collective_BalanceNoCh]
    collective_BalanceYesCh = [a[:smallest_x] for a in collective_BalanceYesCh]
    s = collective_Xs, collective_AUCs, collective_F1s, collective_BalanceNoCh, collective_BalanceYesCh
    plot_data.append(np.asarray(s))


plt.figure(figsize=(7, 7)) # w, h
xs = plot_data[0][0][0]
print(xs)

LEN = 10

for i, item in enumerate(plot_data):
    name = names[i]
    collective_Xs, collective_AUCs, collective_F1s, collective_BalanceNoCh, collective_BalanceYesCh = item

    # first focus on AUC only
    AUCs_means = np.mean(collective_AUCs, 0)
    AUCs_stds = np.std(collective_AUCs, 0)
    print(name)
    
    print("premeans:", len(AUCs_means), AUCs_means)
    print("prestds:", len(AUCs_stds),  AUCs_stds)


    if len(AUCs_means) < LEN:
        AUCs_means = np.pad(AUCs_means, (0,LEN-len(AUCs_means)), mode='constant')
        AUCs_stds = np.pad(AUCs_stds, (0,LEN-len(AUCs_stds)), mode='constant')

    print("means:", len(AUCs_means), AUCs_means)
    print("stds:", len(AUCs_stds),  AUCs_stds)
    #np.pad(m2, (0,9), mode='constant')
    plt.plot(xs, AUCs_means, color=colors_means[i], label=name)
    plt.plot(xs, AUCs_means+AUCs_stds, color=colors_stds[i])
    plt.plot(xs, AUCs_means-AUCs_stds, color=colors_stds[i])

plt.xticks(xs)
plt.axhline(y=0.90016133, color='magenta', linestyle='--', label="Full dataset")

plt.xlabel('number of data samples')
plt.ylabel('AUC')
plt.legend()
plt.ylim(0.0, 1.0)
plt.savefig("[FOOOO]_AUCs.png")
plt.savefig("[FOOOO]_AUCs.pdf")

#plt.show() ### <
plt.close()


# PLOT #2

xs = plot_data[0][0][0]
print(xs)

per_experiment = {}

for i, item in enumerate(plot_data):
    name = names[i]

    collective_Xs, collective_AUCs, collective_F1s, collective_BalanceNoCh, collective_BalanceYesCh = item

    print(name, "|", collective_BalanceNoCh, "|",collective_BalanceYesCh)
    #print(name, "|", collective_BalanceNoCh[0], "|",collective_BalanceYesCh[0])

    print("len(collective_BalanceNoCh)", len(collective_BalanceNoCh), collective_BalanceNoCh.shape)

    noCh_means = np.mean(collective_BalanceNoCh, 0)
    Ch_means = np.mean(collective_BalanceYesCh, 0)
    noCh_stds = np.std(collective_BalanceNoCh, 0)
    Ch_stds = np.std(collective_BalanceYesCh, 0)

    print("noCh_means:", noCh_means)
    print("Ch_means:", Ch_means)
    print("noCh_stds:", noCh_stds)
    print("Ch_stds:", Ch_stds)

    per_experiment[name] = noCh_means, Ch_means, noCh_stds, Ch_stds




print("per_experiment:", per_experiment)

max_y = 1000

#noCh_means = np.mean(col_noCh, axis=1)
#Ch_means = np.mean(col_Ch, axis=1)

for name in names:
    #name = "Random"
    plt.figure(figsize=(4, 4))  # w, h

    noCh_means = per_experiment[name][0]
    Ch_means = per_experiment[name][1]
    noCh_stds = per_experiment[name][2]
    Ch_stds = per_experiment[name][3]


    print("noCh_means:", noCh_means)
    print("Ch_means:", Ch_means)
    print("noCh_stds:", noCh_stds)
    print("Ch_stds:", Ch_stds)


    ind = np.arange(len(Ch_means))

    width = 0.35
    #p1 = plt.bar(ind, Ns_changed, width)
    #p2 = plt.bar(ind, Ns_nochanged, width, bottom=Ns_changed)

    p1 = plt.bar(ind, Ch_means, width, yerr = Ch_stds)
    p2 = plt.bar(ind, noCh_means, width, bottom=Ch_means, yerr = noCh_stds)

    p1.set_label("Change")
    p2.set_label("NoChange")

    plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)
    plt.ylim(0, max_y)
    plt.legend()

    #plt.title(name)
    #plt.xlabel('active learning iterations')
    #plt.ylabel('data sample distribution')
    plt.legend()

    plt.savefig("[FOOOO]_balance_"+name+".png")
    plt.savefig("[FOOOO]_balance_"+name+".pdf")

    #plt.show() ### <
    plt.close()








name = "foooooo"
if False:
    statistics = np.load(to_load)

    pixel_statistics, tile_statistics = statistics
    pixels_thresholds, xs_number_of_data, pixels_recalls, pixels_precisions, pixels_accuracies, pixels_f1s, Ns_changed, Ns_nochanged, pixels_AUCs = pixel_statistics
    tiles_thresholds, xs_number_of_data, tiles_recalls, tiles_precisions, tiles_accuracies, tiles_f1s, Ns_changed, Ns_nochanged = tile_statistics
    
    # keep what we care about together:
    # - AUC score from pixels (pixels_AUCs)
    # - (maybe? f1 from tiles) (tiles_f1s)
    # - balance ( Ns_changed, Ns_nochanged )
    
    
    
    
    
    
    
    # PIXELS
    print("pixels_thresholds:", pixels_thresholds)

    print("xs_number_of_data:", xs_number_of_data)
    print("pixels_recalls:", pixels_recalls)
    print("pixels_precisions:", pixels_precisions)
    print("pixels_accuracies:", pixels_accuracies)
    print("pixels_f1s:", pixels_f1s)
    print("pixels_AUCs:", pixels_AUCs)

    print("Ns_changed:", Ns_changed)
    print("Ns_nochanged:", Ns_nochanged)

    plt.figure(figsize=(7, 7)) # w, h
    plt.plot(xs_number_of_data, pixels_thresholds, color='black', label="thresholds")

    plt.plot(xs_number_of_data, pixels_recalls, color='red', marker='o', label="recalls")
    plt.plot(xs_number_of_data, pixels_precisions, color='blue', marker='o', label="precisions")
    plt.plot(xs_number_of_data, pixels_accuracies, color='green', marker='o', label="accuracies")
    plt.plot(xs_number_of_data, pixels_f1s, color='orange', marker='o', label="f1s")
    plt.plot(xs_number_of_data, pixels_AUCs, color='magenta', marker='o', label="AUCs")

    #plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)

    plt.legend()
    plt.ylim(0.0, 1.0)

    plt.savefig("["+name+"]_dbg_last_al_big_plot_pixelsScores.png")
    plt.close()

    # TILES
    print("tiles_thresholds:", tiles_thresholds)

    print("xs_number_of_data:", xs_number_of_data)
    print("tiles_recalls:", tiles_recalls)
    print("tiles_precisions:", tiles_precisions)
    print("tiles_accuracies:", tiles_accuracies)
    print("tiles_f1s:", tiles_f1s)

    print("Ns_changed:", Ns_changed)
    print("Ns_nochanged:", Ns_nochanged)

    plt.figure(figsize=(7, 7)) # w, h
    plt.plot(xs_number_of_data, tiles_thresholds, color='black', label="thresholds")

    plt.plot(xs_number_of_data, tiles_recalls, color='red', marker='o', label="recalls")
    plt.plot(xs_number_of_data, tiles_precisions, color='blue', marker='o', label="precisions")
    plt.plot(xs_number_of_data, tiles_accuracies, color='green', marker='o', label="accuracies")
    plt.plot(xs_number_of_data, tiles_f1s, color='orange', marker='o', label="f1s")

    #plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)

    plt.legend()
    plt.ylim(0.0, 1.0)

    plt.savefig("["+name+"]_dbg_last_al_big_plot_tilesScores.png")
    plt.close()


    plt.figure(figsize=(7, 7)) # w, h

    N = len(Ns_changed)
    ind = np.arange(N)
    width = 0.35
    p1 = plt.bar(ind, Ns_changed, width)
    p2 = plt.bar(ind, Ns_nochanged, width, bottom=Ns_changed)

    plt.ylabel('number of data samples')
    plt.xticks(np.arange(len(xs_number_of_data)), xs_number_of_data)
    plt.legend((p1[0], p2[0]), ('Change', 'NoChange'))

    plt.savefig("["+name+"]_dbg_last_al_balance_plot.png")
    plt.close()
    
