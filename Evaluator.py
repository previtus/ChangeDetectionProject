from builtins import print

import Debugger
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

class Evaluator(object):
    """
    Class to hold everything related to evaluation of the model's performance.
    """

    def __init__(self, settings):
        self.settings = settings
        self.debugger = Debugger.Debugger(settings)

    def histogram_of_predictions(self, predictions):
        print("We have", len(predictions), "predictions, each is a", predictions[0].shape, "image.")

        flat_predictions = predictions.flatten() # (works for 2D, nD and simple 1D class labels)

        fig = plt.figure()
        bins = 33
        values_of_bins, bins, patches = plt.hist(flat_predictions, bins, facecolor='g', alpha=0.75)
        #plt.yscale('log', nonposy='clip')

        plt.title('Histogram of raw predicted values from the model\n(how much are they around 0.5 vs at the edges)')
        plt.xlabel('Pixel values (0 and 1 being the class categories)')
        plt.ylabel('Number of pixels')

        plt.show()

        #fig = plt.figure()
        #sorted_predictions = sorted(flat_predictions)
        #plt.plot(sorted_predictions)
        #plt.show()

    def try_all_thresholds_per_tiles(self, predicted, labels_orig, range_values = [0.0, 0.5, 1.0], title_txt="", show=True, save=False, name=""):
        import matplotlib.pyplot as plt

        labels = np.array(labels_orig, copy=True)

        test_Tiles = self.mask_label_into_class_label(labels)

        plt.figure(figsize=(10, 3)) # w, h

        xs = []
        ys_recalls = []
        ys_precisions = []
        ys_accuracies = []
        ys_f1s= []
        for thr in range_values: #np.arange(0.0,1.0,0.01):

            predictions_thresholded = np.array(predicted, copy=True)
            for image in predictions_thresholded:
                image[image >= thr] = 1
                image[image < thr] = 0
            predicted_Tiles = self.mask_label_into_class_label(predictions_thresholded)

            #print("test_Tiles>",np.asarray(test_Tiles).shape)
            #print("predicted_Tiles>",np.asarray(predicted_Tiles).shape)

            xs.append(thr)
            print("threshold=",thr)
            #_, recall, precision, accuracy = self.calculate_metrics(predicted, labels, threshold=thr)
            if "NoChange" in title_txt:
                print("from the position of NoChange class instead...")
                recall, precision, accuracy, f1 = self.calculate_recall_precision_accuracy_NOCHANGECLASS(predicted_Tiles, test_Tiles, threshold=thr, need_f1=True)
            else:
                recall, precision, accuracy, f1 = self.calculate_recall_precision_accuracy(predicted_Tiles, test_Tiles, threshold=thr, need_f1=True)

            ys_recalls.append(recall)
            ys_precisions.append(precision)
            ys_accuracies.append(accuracy)
            ys_f1s.append(f1)

        print("xs", len(xs), xs)
        print("ys_recalls", len(ys_recalls), ys_recalls)
        print("ys_precisions", len(ys_precisions), ys_precisions)
        print("ys_accuracies", len(ys_accuracies), ys_accuracies)
        print("ys_f1s", len(ys_f1s), ys_f1s)

        if title_txt == "":
            plt.title('Changing the threshold values')
        else:
            plt.title(title_txt)
        plt.xlabel('threshold value')
        plt.ylabel('metrics')

        plt.plot(xs, ys_recalls, color='red', marker='o', label="Recall")
        plt.plot(xs, ys_precisions, color='blue', marker='o', label="Precision")
        plt.plot(xs, ys_accuracies, color='green', marker='o', label="Accuracy")
        plt.plot(xs, ys_f1s, color='orange', marker='o', label="f1")

        plt.legend()

        plt.ylim(0.0, 1.0)

        if save:
           from matplotlib import pyplot as plt
           plt.savefig(name+'_all_thesholds.png')

        if show:
           plt.show()

        plt.close()

        stats = xs,ys_recalls,ys_precisions,ys_accuracies,ys_f1s
        return stats

    def try_all_thresholds(self, predicted_orig, labels_orig, range_values = [0.0, 0.5, 1.0], title_txt="", show=True, save=False, name=""):

        labels = np.array(labels_orig, copy=True)
        predicted = np.array(predicted_orig, copy=True)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 3)) # w, h

        xs = []
        ys_recalls = []
        ys_precisions = []
        ys_accuracies = []
        ys_f1s= []
        for thr in range_values: #np.arange(0.0,1.0,0.01):
            xs.append(thr)
            print("threshold=",thr)
            #_, recall, precision, accuracy = self.calculate_metrics(predicted, labels, threshold=thr)
            if "NoChange" in title_txt:
                print("from the position of NoChange class instead...")
                recall, precision, accuracy, f1 = self.calculate_recall_precision_accuracy_NOCHANGECLASS(predicted, labels, threshold=thr, need_f1=True)
            else:
                recall, precision, accuracy, f1 = self.calculate_recall_precision_accuracy(predicted, labels, threshold=thr, need_f1=True)

            ys_recalls.append(recall)
            ys_precisions.append(precision)
            ys_accuracies.append(accuracy)
            ys_f1s.append(f1)

        print("xs", len(xs), xs)
        print("ys_recalls", len(ys_recalls), ys_recalls)
        print("ys_precisions", len(ys_precisions), ys_precisions)
        print("ys_accuracies", len(ys_accuracies), ys_accuracies)
        print("ys_f1s", len(ys_f1s), ys_f1s)

        if title_txt == "":
            plt.title('Changing the threshold values')
        else:
            plt.title(title_txt)
        plt.xlabel('threshold value')
        plt.ylabel('metrics')

        plt.plot(xs, ys_recalls, color='red', marker='o', label="Recall")
        plt.plot(xs, ys_precisions, color='blue', marker='o', label="Precision")
        plt.plot(xs, ys_accuracies, color='green', marker='o', label="Accuracy")
        plt.plot(xs, ys_f1s, color='orange', marker='o', label="f1")

        plt.legend()

        plt.ylim(0.0, 1.0)

        if save:
           from matplotlib import pyplot as plt
           plt.savefig(name+'_all_thesholds.png')

        if show:
           plt.show()

        plt.close()

        stats = xs,ys_recalls,ys_precisions,ys_accuracies,ys_f1s
        return stats

    def text_report(self, predictions_orig, ground_truths_orig, threshold, save_text_file="", as_tiles = False):
        predictions = np.array(predictions_orig, copy=True)
        ground_truths = np.array(ground_truths_orig, copy=True)

        if as_tiles:
            ground_truths = self.mask_label_into_class_label(ground_truths)
            predictions_thresholded = np.array(predictions)
            for image in predictions_thresholded:
                image[image >= threshold] = 1
                image[image < threshold] = 0
            predictions_copy = self.mask_label_into_class_label(predictions_thresholded)
        else:
            if len(predictions.shape) > 1:
                predictions_copy = np.array(predictions)
            else:
                predictions_copy = np.array([predictions])

            for image in predictions_copy:
                image[image >= threshold] = 1
                image[image < threshold] = 0

        arr_predictions = predictions_copy.flatten()
        arr_gts = ground_truths.flatten()

        sklearn_accuracy = sklearn.metrics.accuracy_score(arr_gts, arr_predictions)
        sklearn_precision = sklearn.metrics.precision_score(arr_gts, arr_predictions)
        sklearn_recall = sklearn.metrics.recall_score(arr_gts, arr_predictions)
        sklearn_f1 = sklearn.metrics.f1_score(arr_gts, arr_predictions)
        stats_str = "Stats: acc "+str(sklearn_accuracy)+", prec "+str(sklearn_precision)+", recall "+str(sklearn_recall)+", f1 "+str(sklearn_f1)
        print(stats_str)
        labels = ["no change", "change"] # 0 no change, 1 change
        report = str(sklearn.metrics.classification_report(arr_gts, arr_predictions, target_names=labels))
        print(report)
        conf = sklearn.metrics.confusion_matrix(arr_gts, arr_predictions)
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
        TruePositiveRate = TP / (TP+FN)
        FalsePositiveRate = FP / (FP+TN)

        conf_str += "TruePositiveRate = TP / (TP+FN) = "+str(TruePositiveRate)+"\n"
        conf_str += "FalsePositiveRate = FP / (FP+TN) = "+str(FalsePositiveRate)+"\n"

        print(conf_str)


        if save_text_file is not "":
            text_report = "Using threshold "+str(threshold)+" we get:\n"+report
            text_report += "\n"
            text_report += str(conf_str)
            text_report += "\n\n"+stats_str
            file = open(save_text_file, "w")
            file.write(text_report)
            file.close()


    def calculate_f1(self, predictions, ground_truths, threshold = 0.5):
        if len(predictions.shape) > 1:
            predictions_copy = np.array(predictions)
        else:
            predictions_copy = np.array([predictions])

        for image in predictions_copy:
            image[image >= threshold] = 1
            image[image < threshold] = 0

        arr_predictions = predictions_copy.flatten()
        arr_gts = ground_truths.flatten()

        sklearn_f1 = sklearn.metrics.f1_score(arr_gts, arr_predictions)

        return sklearn_f1

    def calculate_auc_roc(self, predictions, ground_truths, name):
        # PS: arr_predictions might be needed non-thresholded!
        # performance of a binary classifier system as its discrimination threshold is varied
        unthresholded = predictions.flatten()
        arr_gts = ground_truths.flatten()

        auc = sklearn.metrics.roc_auc_score(arr_gts, unthresholded)
        # ROC AUC varies between 0 and 1 — with an uninformative classifier yielding 0.5

        # or a plot
        # sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(arr_gts, unthresholded, pos_label=None, sample_weight=None,
                                                         drop_intermediate=True)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(name+"ROC_curveWith_AUC.png")
        plt.close()

        return auc


    def calculate_recall_precision_accuracy(self, predictions, ground_truths, threshold = 0.5, need_f1=False, need_auc=False, save_text_file=""):
        if len(predictions.shape) > 1:
            predictions_copy = np.array(predictions)
        else:
            predictions_copy = np.array([predictions])

        for image in predictions_copy:
            image[image >= threshold] = 1
            image[image < threshold] = 0

        arr_predictions = predictions_copy.flatten()
        arr_gts = ground_truths.flatten()

        sklearn_accuracy = sklearn.metrics.accuracy_score(arr_gts, arr_predictions)
        sklearn_precision = sklearn.metrics.precision_score(arr_gts, arr_predictions)
        sklearn_recall = sklearn.metrics.recall_score(arr_gts, arr_predictions)

        sklearn_f1 = 0.0
        if need_f1:
            sklearn_f1 = sklearn.metrics.f1_score(arr_gts, arr_predictions)

        if save_text_file is not "":
            labels = ["no change", "change"]  # 0 no change, 1 change
            text_report = str(sklearn.metrics.classification_report(arr_gts, arr_predictions, target_names=labels))
            text_report += "\n"
            text_report += str(sklearn.metrics.confusion_matrix(arr_gts, arr_predictions))
            file = open(save_text_file, "w")
            file.write(text_report)
            file.close()

        return sklearn_recall, sklearn_precision, sklearn_accuracy, sklearn_f1

    def calculate_recall_precision_accuracy_NOCHANGECLASS(self, predictions, ground_truths, threshold = 0.5):
        if len(predictions.shape) > 1:
            predictions_copy = np.array(predictions)
        else:
            predictions_copy = np.array([predictions])

        for image in predictions_copy:
            image[image >= threshold] = 1
            image[image < threshold] = 0

        arr_predictions = predictions_copy.flatten()
        arr_gts = ground_truths.flatten()

        sklearn_accuracy = sklearn.metrics.accuracy_score(arr_gts, arr_predictions)
        sklearn_precision = sklearn.metrics.precision_score(arr_gts, arr_predictions, pos_label=0) # NO CHANGE CLASS
        sklearn_recall = sklearn.metrics.recall_score(arr_gts, arr_predictions, pos_label=0) # NO CHANGE CLASS
        sklearn_f1 = sklearn.metrics.f1_score(arr_gts, arr_predictions)

        return sklearn_recall, sklearn_precision, sklearn_accuracy, sklearn_f1

    def calculate_metrics(self, predictions, ground_truths, threshold = 0.5, verbose=2, save_text_file=""):

        flavour_text = ""
        if len(predictions.shape) > 1:
            if verbose >= 2:
                print("We have", len(predictions), "predictions, each is a", predictions[0].shape, "image.", predictions[0][0][0:3])
                print("We have", len(ground_truths), "ground truths, each is a", ground_truths[0].shape, "image.", ground_truths[0][0][0:3])
            flavour_text = "pixels"
            # careful not to edit the label images here
            predictions_copy = np.array(predictions)
        else:
            flavour_text = "labels"
            predictions_copy = np.array([predictions])

        # 1 threshold the data per each pixel

        # Sith thinks in absolutes
        for image in predictions_copy:
            image[image >= threshold] = 1
            image[image < threshold] = 0

        # only "0.0" and "1.0" in the data now

        #if True:
        #    print("pred:",predictions_copy[0].astype(int))
        #    print("gt:  ",ground_truths)

        # 2 calculate T/F P/N

        arr_predictions = predictions_copy.flatten()
        arr_gts = ground_truths.flatten()

        #print("We have", len(arr_predictions), "~", len(arr_gts), "pixels.")
        assert len(arr_predictions) == len(arr_gts)

        FN = 0
        FP = 0
        TP = 0
        TN = 0

        # from the standpoint of the "changed" (1.0) class:
        for pixel_i in range(len(arr_predictions)):
            pred = arr_predictions[pixel_i]
            gt = arr_gts[pixel_i]

            if pred == 0.0 and gt == 0.0:
                TN += 1
            elif pred == 1.0 and gt == 1.0:
                TP += 1
            elif pred == 1.0 and gt == 0.0:
                FP += 1
            elif pred == 0.0 and gt == 1.0:
                FN += 1

        total = FP + FN + TP + TN

        # 3a generate confusion matrix
        # 3b metrics - recall, precision, accuracy

        if verbose >= 2:
            print("Statistics over", total,flavour_text,":")
            print("TP", TP, "\t ... correctly classified as a change.")
            print("TN", TN, "\t ... correctly classified as a no-change.")
            print("FP", FP, "\t ... classified as change while it's not.")
            print("FN", FN, "\t ... classified as no-change while it is one.")

        accuracy = float(TP + TN) / float(total)
        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)

        print("accuracy", accuracy, "\t")
        print("precision", precision, "\t")
        print("recall", recall, "\t")

        # 3b metrics - IoU
        IoU = float(TP) / float(TP + FP + FN)
        print("IoU", IoU)

        #sklearn_precision = sklearn.metrics.precision_score(arr_gts, arr_predictions)
        #sklearn_recall = sklearn.metrics.recall_score(arr_gts, arr_predictions)
        #print("sklearn_precision", sklearn_precision, "\t")
        #print("sklearn_recall", sklearn_recall, "\t")

        sklearn_f1 = sklearn.metrics.f1_score(arr_gts, arr_predictions)
        print("sklearn_f1", sklearn_f1, "\t")

        labels = ["no change", "change"] # 0 no change, 1 change
        if verbose >= 2:
            print(sklearn.metrics.classification_report(arr_gts, arr_predictions, target_names=labels))
            conf = sklearn.metrics.confusion_matrix(arr_gts, arr_predictions)
            print(conf)

            if save_text_file is not "":
                text_report = str(sklearn.metrics.classification_report(arr_gts, arr_predictions, target_names=labels))
                text_report += "\n"
                text_report += str(conf)
                file = open(save_text_file, "w")
                file.write(text_report)
                file.close()

            print("=====================================================================================")

        predictions_thresholded = predictions_copy
        return predictions_thresholded, recall, precision, accuracy, sklearn_f1


    # chopped out some unnecessary things:
    def calculate_metrics_fast(self, predictions, ground_truths, threshold = 0.5, verbose=2):

        flavour_text = ""
        if len(predictions.shape) > 1:
            if verbose >= 2:
                print("We have", len(predictions), "predictions, each is a", predictions[0].shape, "image.", predictions[0][0][0:3])
                print("We have", len(ground_truths), "ground truths, each is a", ground_truths[0].shape, "image.", ground_truths[0][0][0:3])
            flavour_text = "pixels"
            # careful not to edit the label images here
            predictions_copy = np.array(predictions)
        else:
            flavour_text = "labels"
            predictions_copy = np.array([predictions])

        # 1 threshold the data per each pixel

        # Sith thinks in absolutes
        for image in predictions_copy:
            image[image >= threshold] = 1
            image[image < threshold] = 0

        # only "0.0" and "1.0" in the data now

        #if True:
        #    print("pred:",predictions_copy[0].astype(int))
        #    print("gt:  ",ground_truths)

        # 2 calculate T/F P/N

        arr_predictions = predictions_copy.flatten()
        arr_gts = ground_truths.flatten()

        #print("We have", len(arr_predictions), "~", len(arr_gts), "pixels.")
        assert len(arr_predictions) == len(arr_gts)

        FN = 0
        FP = 0
        TP = 0
        TN = 0

        # from the standpoint of the "changed" (1.0) class:
        for pixel_i in range(len(arr_predictions)):
            pred = arr_predictions[pixel_i]
            gt = arr_gts[pixel_i]

            if pred == 0.0 and gt == 0.0:
                TN += 1
            elif pred == 1.0 and gt == 1.0:
                TP += 1
            elif pred == 1.0 and gt == 0.0:
                FP += 1
            elif pred == 0.0 and gt == 1.0:
                FN += 1

        total = FP + FN + TP + TN

        # 3a generate confusion matrix
        # 3b metrics - recall, precision, accuracy

        if verbose >= 2:
            print("Statistics over", total,flavour_text,":")
            print("TP", TP, "\t ... correctly classified as a change.")
            print("TN", TN, "\t ... correctly classified as a no-change.")
            print("FP", FP, "\t ... classified as change while it's not.")
            print("FN", FN, "\t ... classified as no-change while it is one.")

        accuracy = float(TP + TN) / float(total)
        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)

        print("accuracy", accuracy, "\t")
        print("precision", precision, "\t")
        print("recall", recall, "\t")

        predictions_thresholded = predictions_copy
        return predictions_thresholded, recall, precision, accuracy

    # select thr which maximizes the f1 score
    def metrics_autothr_f1_max(self, predictions, ground_truths, jump_by = 0.1, save_text_file=""):
        # force it selecting something 'sensible' for the threshold ...
        range_values = np.arange(0.0+jump_by, 1.0, jump_by)

        xs = []
        ys_recalls = []
        ys_precisions = []
        ys_accuracies = []
        ys_f1s = []
        for thr in range_values:
            xs.append(thr)
            print("auto threshold=", thr)

            f1 = self.calculate_f1(predictions, ground_truths, threshold=thr)
            ys_f1s.append(f1)

        max_f1_idx = np.argmax(ys_f1s)
        best_thr = xs[max_f1_idx]

        selected_recall, selected_precision, selected_accuracy, _ = self.calculate_recall_precision_accuracy(predictions, ground_truths,threshold=thr, need_f1=False, save_text_file=save_text_file)
        selected_f1 = ys_f1s[max_f1_idx]

        print("Selecting threshold as", best_thr, "as it maximizes the f1 score getting", selected_f1,
              "(other scores are: recall", selected_recall, ", precision", selected_precision, ", acc", selected_accuracy, ")")

        return best_thr, selected_recall, selected_precision, selected_accuracy, selected_f1


    def mask_label_into_class_label(self, mask_labels, img_resolution = 256, bigger_than_percent=3.0):
        """
        Converts the mask label images (for example 224x224 pixel image with 0s and 1s) into a single class label
        ("change" or "no change") using the same threshold as when balancing the data.
        PS: we could use different threshold here ...
        Slight problem is that we won't be exactly sure that the "change" is really "change" and not just noisy
        mask label (to do: clean label data)

        :param mask_labels:
        :return:
        """
        array_of_number_of_change_pixels = []

        for mask in mask_labels:
            number_of_ones = np.count_nonzero(mask.flatten()) # << loading takes care of this 0 vs non-zero
            array_of_number_of_change_pixels.append(number_of_ones)

        self.debugger.save_arr(array_of_number_of_change_pixels, "BALANCING")
        array_of_number_of_change_pixels = self.debugger.load_arr("BALANCING")

        array_of_number_of_change_pixels = array_of_number_of_change_pixels / (
                img_resolution * img_resolution) * 100.0  # percentage of image changed

        class_labels = []
        for value in array_of_number_of_change_pixels:
            is_change = value > bigger_than_percent
            class_labels.append(int(is_change))

        return np.array(class_labels)


    def human_legible_tiles_report(self, predicted_orig, labels_orig, wanted_recall, recalls, thresholds):

        labels = np.array(labels_orig, copy=True)
        test_Tiles = self.mask_label_into_class_label(labels)
        arr_gts = test_Tiles.flatten()

        N = len(arr_gts)
        # worst case scenario:
        best_recall_cost = N
        best_recall_idx = 0

        for i, thr in reversed(list(enumerate(thresholds))):

            r = recalls[i]
            if r > wanted_recall:
                # cost = how many tiles we have to check = TP+FP

                # for i, thr in enumerate(thresholds):
                # recomputing theses scores here seems wasteful ...
                predictions_thresholded = np.array(predicted_orig, copy=True)
                for image in predictions_thresholded:
                    image[image >= thr] = 1
                    image[image < thr] = 0
                predicted_Tiles = self.mask_label_into_class_label(predictions_thresholded)
                arr_predictions = predicted_Tiles.flatten()
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

        report_str = "If we want the recall to be better than "+str(wanted_recall)+\
                     ", we need to set the threshold to be = "+str(thresholds[best_recall_idx])+" which will give us " \
                     "recall of "+str(recalls[best_recall_idx])+" while the number of tiles needed to check is "+\
                     str(best_recall_cost)+" from the worst case scenario "+str(N)+" (that's "+str(np.round(100*(best_recall_cost/N), 2))+"%).\n\n"

        print(report_str)
        return report_str

    # ================= Unified test func call:

    def unified_test_report(self, models, testing_set, validation_set, postprocessor, name, threshold_fineness = 0.05, optionally_save_missclassified = False, optional_manual_exclusions=[], optional_additional_predAndGts = []):
        if len(models) > 1:
            print("Testing model ensemble:", len(models), "*" ,models[0],"on test set (size", len(testing_set[0]),")")
        else:
            print("Testing model:", models[0], "on test set (size", len(testing_set[0]), ")")

        test_L, test_R, test_V = testing_set

        if validation_set is not None:
            val_L, val_R, val_V = validation_set
            if val_L.shape[3] > 3:
                # 3 channels only - rgb
                val_L = val_L[:, :, :, 1:4]
                val_R = val_R[:, :, :, 1:4]

        if test_L.shape[3] > 3:
            # 3 channels only - rgb
            test_L = test_L[:,:,:,1:4]
            test_R = test_R[:,:,:,1:4]

        if len(models) > 1:
            ensemble_predictions = []
            for model in models:
                print("predicting for the test set")
                predicted = model.predict(x=[test_L, test_R], batch_size=4)
                ensemble_predictions.append(predicted)

            if validation_set is not None:
                ensemble_val_predictions = []
                for model in models:
                    print("predicting for the val set")
                    predicted_val = model.predict(x=[val_L, val_R], batch_size=4)
                    ensemble_val_predictions.append(predicted_val)


            print("TODO: mean from the predictions of an ensemble")
            print("HAX, use just the 1st model")
            predicted = ensemble_predictions[0]


            predicted_ITHINK = np.mean(ensemble_predictions, axis=0)
            print("predicted_ITHINK.shape", predicted_ITHINK.shape, "should be the same as", predicted.shape)
            print("first pixels")
            for i in range(len(ensemble_predictions)):
                print(ensemble_predictions[i][0][0])
            print("avg into")
            print(predicted_ITHINK[0][0][0])
            print("right? (they should!)")

        else:
            predicted = models[0].predict(x=[test_L, test_R], batch_size=4)

            if validation_set is not None:
                predicted_val = models[0].predict(x=[val_L, val_R], batch_size=4)

        # with just 2 classes I can hax:
        predicted = predicted[:, :, :, 1]
        predicted = postprocessor.postprocess_labels(predicted)

        if validation_set is not None:
            predicted_val = predicted_val[:, :, :, 1]
            predicted_val = postprocessor.postprocess_labels(predicted_val)

        officially_we_have_N = len(predicted)

        if len(optional_manual_exclusions) > 0:
            ### Will have to redo for the 10foldcrossval if I get the test set differently
            ### Validation set can stay ...

            # HAXES HEXES:
            len_one = len(predicted)
            good_indices = []
            #print(len(predicted), len(test_V), len(test_L), len(test_R))

            for i in range(len(predicted)):
                if i not in optional_manual_exclusions:
                    good_indices.append(i)

            predicted = [predicted[i] for i in good_indices]
            test_V = [test_V[i] for i in good_indices]

            test_L = [test_L[i] for i in good_indices]
            test_R = [test_R[i] for i in good_indices]

            predicted = np.asarray(predicted)
            test_V = np.asarray(test_V)
            test_L = np.asarray(test_L)
            test_R = np.asarray(test_R)

            print("Exclusion of incorrect labels from the set - from", len_one, "to", len(predicted))
            officially_we_have_N = len(predicted)

        if len(optional_additional_predAndGts) > 0:
            additional_predicted, additional_gts = optional_additional_predAndGts

            print("predicted.shape", predicted.shape)
            print("additional_predicted.shape", additional_predicted.shape)

            predicted = np.append(predicted, additional_predicted, 0)
            test_V = np.append(test_V, additional_gts, 0)
            print("after appending predicted.shape", predicted.shape)
            print("after appending (gts) test_V.shape", test_V.shape)

        # Unified reporting:
        # - 1.) evaluation per pixel
        # --- test all thresholds, save image
        # --- select best (f1)
        # --- save text output and human-legible report

        ### ??? Per pixel AUC:
        pixels_auc = self.calculate_auc_roc(predicted, test_V, name=name) # < could the ROC curve tell us which thr to choose?? In that case I'd call that on the validation val_V (if I had it)

        show = False
        save = True

        print("::: PER PIXEL EVALUATION :::")
        # range should include the end points (0.0 and 1.0)
        # np.arange(0.0+jump_by, 1.0, jump_by) - without the corners
        # np.arange(0.0, 1.0+jump_by, jump_by) - with the corners
        pixels_stats = self.try_all_thresholds(predicted, test_V, np.arange(0.0, 1.0+threshold_fineness, threshold_fineness),
                                     title_txt="Masks (all pixels 0/1) evaluated [Change Class]",
                                     show=show, save=save, name=name+"Pixels")
        pixels_xs_tresholds, pixels_ys_recalls, pixels_ys_precisions, pixels_ys_accuracies, pixels_ys_f1s = pixels_stats

        print("xs_tresholds",pixels_xs_tresholds)
        print("ys_recalls",pixels_ys_recalls)
        print("ys_precisions",pixels_ys_precisions)
        print("ys_accuracies",pixels_ys_accuracies)
        print("ys_f1s",pixels_ys_f1s)
        # for maximum we don't allow the end points thought
        pixels_max_f1_idx = np.argmax(pixels_ys_f1s[1:-1]) + 1
        pixels_best_thr = pixels_xs_tresholds[pixels_max_f1_idx]

        note_txt = ""

        # Make this decision on Validation set!
        if validation_set is not None:
            print("All thrs on validation set:")
            val_pixels_auc = self.calculate_auc_roc(predicted_val, val_V, name=name+"__onValidationSet__")

            validation_pixels_stats = self.try_all_thresholds(predicted_val, val_V,
                                                   np.arange(0.0, 1.0 + threshold_fineness, threshold_fineness),
                                                   title_txt="Masks (all pixels 0/1) evaluated [Change Class] on ValidationSet",
                                                   show=show, save=save, name=name + "PixelsVAL")
            val_pixels_xs_tresholds, val_pixels_ys_recalls, val_pixels_ys_precisions, val_pixels_ys_accuracies, val_pixels_ys_f1s = validation_pixels_stats

            print("val_pixels_xs_tresholds", val_pixels_xs_tresholds)
            print("val_pixels_ys_recalls", val_pixels_ys_recalls)
            print("val_pixels_ys_precisions", val_pixels_ys_precisions)
            print("val_pixels_ys_accuracies", val_pixels_ys_accuracies)
            print("val_pixels_ys_f1s", val_pixels_ys_f1s)
            print("val_pixels_auc", val_pixels_auc)
            # for maximum we don't allow the end points thought
            val_pixels_max_f1_idx = np.argmax(val_pixels_ys_f1s[1:-1]) + 1
            val_pixels_best_thr = val_pixels_xs_tresholds[val_pixels_max_f1_idx]

            pixels_best_thr = val_pixels_best_thr
            pixels_max_f1_idx = val_pixels_max_f1_idx

            note_txt = "(on the Validation set)"

        pixels_selected_recall = pixels_ys_recalls[pixels_max_f1_idx]
        pixels_selected_precision = pixels_ys_precisions[pixels_max_f1_idx]
        pixels_selected_accuracy = pixels_ys_accuracies[pixels_max_f1_idx]
        pixels_selected_f1 = pixels_ys_f1s[pixels_max_f1_idx]


        print("Per pixel - Selecting threshold as", pixels_best_thr, "as it maximizes the f1 score "+note_txt+" getting", pixels_selected_f1,
              "(other scores are: recall", pixels_selected_recall, ", precision", pixels_selected_precision, ", acc", pixels_selected_accuracy, ")")

        # text outputs for the best setting:
        self.text_report(predicted, test_V, pixels_best_thr, save_text_file=name+"Pixels.txt", as_tiles=False)

        print("=====================================================================================")


        # - 2.) evaluation per tile
        # --- test all thresholds, save image
        # --- select best (f1)
        # --- save text output and human-legible report
        # --- (optionally) save missclassified images

        print("::: PER TILE EVALUATION :::")
        tiles_stats = self.try_all_thresholds_per_tiles(predicted, test_V, np.arange(0.0, 1.0+threshold_fineness, threshold_fineness),
                                     title_txt="Tiles (tile 0/1) evaluated [Change Class]",
                                     show=show, save=save, name=name+"Tiles")

        tiles_xs_tresholds, tiles_ys_recalls, tiles_ys_precisions, tiles_ys_accuracies, tiles_ys_f1s = tiles_stats

        print("xs_tresholds",tiles_xs_tresholds)
        print("ys_recalls",tiles_ys_recalls)
        print("ys_precisions",tiles_ys_precisions)
        print("ys_accuracies",tiles_ys_accuracies)
        print("ys_f1s",tiles_ys_f1s)

        tiles_max_f1_idx = np.argmax(tiles_ys_f1s[1:-1]) + 1
        tiles_best_thr = tiles_xs_tresholds[tiles_max_f1_idx]

        note_txt = ""

        # Make this decision on Validation set!
        if validation_set is not None:
            print("All thrs on validation set:")
            validation_tiles_stats = self.try_all_thresholds_per_tiles(predicted_val, val_V,
                                                   np.arange(0.0, 1.0 + threshold_fineness, threshold_fineness),
                                                   title_txt="Tiles (tile 0/1) evaluated [Change Class] on ValidationSet",
                                                   show=show, save=save, name=name + "TilesVAL")
            val_tiles_xs_tresholds, val_tiles_ys_recalls, val_tiles_ys_precisions, val_tiles_ys_accuracies, val_tiles_ys_f1s = validation_tiles_stats

            print("val_tiles_xs_tresholds", val_tiles_xs_tresholds)
            print("val_tiles_ys_recalls", val_tiles_ys_recalls)
            print("val_tiles_ys_precisions", val_tiles_ys_precisions)
            print("val_tiles_ys_accuracies", val_tiles_ys_accuracies)
            print("val_tiles_ys_f1s", val_tiles_ys_f1s)

            # for maximum we don't allow the end points thought
            val_tiles_max_f1_idx = np.argmax(val_tiles_ys_f1s[1:-1]) + 1
            val_tiles_best_thr = val_tiles_xs_tresholds[val_tiles_max_f1_idx]

            print("] Per tile on validation set - we select threshold as", val_tiles_best_thr,
                  "as it maximizes the val f1 score getting", val_tiles_ys_f1s[val_tiles_max_f1_idx],
                  "(other scores are: val recall", val_tiles_ys_recalls[val_tiles_max_f1_idx], ", val precision", val_tiles_ys_precisions[val_tiles_max_f1_idx], ", val acc",
                  val_tiles_ys_accuracies[val_tiles_max_f1_idx], ")")

            tiles_best_thr = val_tiles_best_thr
            tiles_max_f1_idx = val_tiles_max_f1_idx

            note_txt = "(on the Validation set)"


        tiles_selected_recall = tiles_ys_recalls[tiles_max_f1_idx]
        tiles_selected_precision = tiles_ys_precisions[tiles_max_f1_idx]
        tiles_selected_accuracy = tiles_ys_accuracies[tiles_max_f1_idx]
        tiles_selected_f1 = tiles_ys_f1s[tiles_max_f1_idx]

        print("Per tile - Selecting threshold as", tiles_best_thr, "as it maximizes the f1 score "+note_txt+" getting", tiles_selected_f1,
              "(other scores are: recall", tiles_selected_recall, ", precision", tiles_selected_precision, ", acc", tiles_selected_accuracy, ")")

        # text report for per tile eval

        self.text_report(predicted, test_V, tiles_best_thr, save_text_file=name + "Tiles.txt", as_tiles=True)

        wanted_txt = ""
        wanted_recall = 0.1
        wanted_txt += self.human_legible_tiles_report(predicted, test_V, wanted_recall, tiles_ys_recalls, tiles_xs_tresholds)
        wanted_recall = 0.3
        wanted_txt += self.human_legible_tiles_report(predicted, test_V, wanted_recall, tiles_ys_recalls, tiles_xs_tresholds)
        wanted_recall = 0.5
        wanted_txt += self.human_legible_tiles_report(predicted, test_V, wanted_recall, tiles_ys_recalls, tiles_xs_tresholds)
        wanted_recall = 0.7
        wanted_txt += self.human_legible_tiles_report(predicted, test_V, wanted_recall, tiles_ys_recalls, tiles_xs_tresholds)
        wanted_recall = 0.75
        wanted_txt += self.human_legible_tiles_report(predicted, test_V, wanted_recall, tiles_ys_recalls, tiles_xs_tresholds)
        wanted_recall = 0.8
        wanted_txt += self.human_legible_tiles_report(predicted, test_V, wanted_recall, tiles_ys_recalls, tiles_xs_tresholds)
        wanted_recall = 0.85
        wanted_txt += self.human_legible_tiles_report(predicted, test_V, wanted_recall, tiles_ys_recalls, tiles_xs_tresholds)
        wanted_recall = 0.9
        wanted_txt += self.human_legible_tiles_report(predicted, test_V, wanted_recall, tiles_ys_recalls, tiles_xs_tresholds)
        wanted_recall = 0.95
        wanted_txt += self.human_legible_tiles_report(predicted, test_V, wanted_recall, tiles_ys_recalls, tiles_xs_tresholds)

        file = open(name + "HumanLegible.txt", "w")
        file.write(wanted_txt)
        file.close()



        # save missclassifications (optionally)

        if optionally_save_missclassified:
            threshold = tiles_best_thr
            ground_truths = np.array(test_V, copy=True)
            test_classlabels = self.mask_label_into_class_label(ground_truths)
            predictions_thresholded = np.array(predicted, copy=True)
            for image in predictions_thresholded:
                image[image >= threshold] = 1
                image[image < threshold] = 0
            predicted_classlabels = self.mask_label_into_class_label(predictions_thresholded)

            # Get indices of the misclassified samples
            misclassified_indices = np.where(predicted_classlabels != test_classlabels)
            misclassified_indices = misclassified_indices[0]

            text_to_save_missclassifieds = "From "+str(officially_we_have_N)+"samples in the original test set (with corrections)\n"
            print("misclassified_indices:", misclassified_indices)
            text_to_save_missclassifieds += "misclassified_indices:" + str(misclassified_indices) + "\n"
            for ind in misclassified_indices:
                # print("idx", ind, ":", predicted_classlabels[ind]," != ",test_classlabels[ind])
                text_to_save_missclassifieds += "idx " + str(ind) + ": " + str(
                    predicted_classlabels[ind]) + " != " + str(test_classlabels[ind]) + "\n"

            path = name + "MissedIndices.txt"
            file = open(path, "w")
            file.write(text_to_save_missclassifieds)
            file.close()

            test_L, test_R = postprocessor.postprocess_images(test_L, test_R)

            if test_L.shape[3] > 3:
                # 3 channels only - rgb
                test_L = test_L[:, :, :, 1:4]
                test_R = test_R[:, :, :, 1:4]

            if len(optional_additional_predAndGts) > 0:
                # remove those indices which we don't have really loaded
                misclassified_indices = [i for i in misclassified_indices if i <= officially_we_have_N]

            print("Misclassified samples (in total", len(misclassified_indices), "):")
            off = 0
            by = 4
            by = min(by, len(misclassified_indices))
            while off < len(misclassified_indices):
                by_rem = min(by, len(misclassified_indices) - off)

                # self.debugger.viewTripples(test_L, test_R, test_V, how_many=4, off=off)
                self.debugger.viewQuadrupples(test_L[misclassified_indices], test_R[misclassified_indices],
                                              test_V[misclassified_indices], predicted[misclassified_indices],
                                              how_many=by_rem, off=off, show=show, save=save,
                                              name=name + "_missclassified_" + str(off), show_txts=False)
                off += by



            # Also some correctly classified ones pls:
            off = 0
            by = 4
            by = min(by, len(test_L))
            until_n = min(by*10, len(test_L))
            while off < until_n:
                by_rem = min(by, until_n - off)

                self.debugger.viewQuadrupples(test_L, test_R, test_V, predicted, how_many=by_rem, off=off, show=show,save=save,
                                              name=name + "_randomlyselected_" + str(off), show_txts=False)
                off += by


        tiles_stats = tiles_best_thr, tiles_selected_recall, tiles_selected_precision, tiles_selected_accuracy, tiles_selected_f1
        mask_stats = pixels_best_thr, pixels_selected_recall, pixels_selected_precision, pixels_selected_accuracy, pixels_selected_f1, pixels_auc
        statistics = mask_stats, tiles_stats
        return statistics


        #return tiles_selected_recall, tiles_selected_precision, tiles_selected_accuracy, tiles_selected_f1, tiles_best_thr

