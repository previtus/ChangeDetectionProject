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

        flat_predictions = predictions.flatten()

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


    def calculate_metrics(self, predictions, ground_truths, threshold = 0.5):
        print("We have", len(predictions), "predictions, each is a", predictions[0].shape, "image.", predictions[0][0][0:3])
        print("We have", len(ground_truths), "ground truths, each is a", ground_truths[0].shape, "image.", ground_truths[0][0][0:3])

        # careful not to edit the label images here
        predictions_copy = np.array(predictions)

        # 1 threshold the data per each pixel

        # Sith thinks in absolutes
        for image in predictions_copy:
            image[image >= threshold] = 1
            image[image < threshold] = 0

        # only "0.0" and "1.0" in the data now

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

        print("Statistics over", total,"pixels:")
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

        sklearn_precision = sklearn.metrics.precision_score(arr_gts, arr_predictions)
        sklearn_recall = sklearn.metrics.recall_score(arr_gts, arr_predictions)
        print("sklearn_precision", sklearn_precision, "\t")
        print("sklearn_recall", sklearn_recall, "\t")

        sklearn_f1 = sklearn.metrics.f1_score(arr_gts, arr_predictions)
        print("sklearn_f1", sklearn_f1, "\t")

        labels = ["no change", "change"] # 0 no change, 1 change

        print(sklearn.metrics.classification_report(arr_gts, arr_predictions, target_names=labels))

        conf = sklearn.metrics.confusion_matrix(arr_gts, arr_predictions)
        print(conf)

        print("=====================================================================================")