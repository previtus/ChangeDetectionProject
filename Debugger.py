import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random
import os
from tqdm import tqdm
import h5py

class Debugger(object):
    """
    Will have functions useful for debugging.
    """



    def __init__(self, settings):
        self.settings = settings
        a = 0


    ### DATASET VISUALIZATIONS:

    #def dynamicRangeInSet(self, set_of_images):
    #    return 0

    def dynamicRangeInImage(self, image):
        ranges = ""
        if len(image.shape) > 2:
            for channel in range(image.shape[2]):
                min_val = np.round(np.min(image[:,:,channel]), 3)
                max_val = np.round(np.max(image[:,:,channel]), 3)
                #min_val = np.min(image[:,:,channel])
                #max_val = np.max(image[:,:,channel])
                ranges += str(min_val)+"-"+str(max_val)+", "
        else:
            ranges += str(np.min(image))+"-"+str(np.max(image))
        return ranges

    def occurancesInImage(self, image):
        values_dict = {}
        for val in image.flatten():
            if val in values_dict:
                values_dict[val] += 1
            else:
                values_dict[val] = 1

        return values_dict

    # maybe also show avg value for labels? - to compare label<->predicted
    def viewVectors(self, images, labels=[], how_many=6, off=0):

        rows, columns = 2, 3
        #fig = plt.figure(figsize=(10, 8))
        fig = plt.figure()
        k = 1
        for idx in range(how_many):

            label = images[idx+off]
            fig.add_subplot(rows, columns, k)
            plt.imshow(label, cmap='gray')

            text = ""
            if len(labels)>0:
                text += str(round(labels[idx+off], 2))+"%"
            #text += "Label shape "+str(label.shape)+"\n"+self.dynamicRangeInImage(label)
            #text = ""
            fig.gca().set(xlabel=text, xticks=[], yticks=[])
            k += 1

        plt.show()
        # also show dimensions, channels, dynamic range of each, occurances in the label (0, 1)


    def viewTripples(self, lefts, rights, labels, how_many=3, off=0):
        #for i in range(len(lefts)):
        #    print(i, "=>", lefts[i].shape, rights[i].shape, labels[i].shape)

        rows, columns = how_many, 3
        fig = plt.figure(figsize=(10, 8))
        k = 1
        for i in range(how_many):
            idx = i #+ random.randint(1, len(lefts)-how_many-off)

            left = lefts[idx+off]
            fig.add_subplot(rows, columns, k)
            if left.shape[2] > 3:
                plt.imshow(left[:,:,1:4])
            else:
                plt.imshow(left)
            text = "Left shape "+str(left.shape)+"\n"+self.dynamicRangeInImage(left)[0:-2]
            fig.gca().set(xlabel=text, xticks=[], yticks=[])

            right = rights[idx+off]
            fig.add_subplot(rows, columns, k+1)
            if right.shape[2] > 3:
                plt.imshow(right[:,:,1:4])
            else:
                plt.imshow(right)

            text = "Right shape "+str(right.shape)+"\n"+self.dynamicRangeInImage(right)[0:-2]
            fig.gca().set(xlabel=text, xticks=[], yticks=[])

            label = labels[idx+off]
            fig.add_subplot(rows, columns, k+2)
            #plt.imshow(label, cmap='gray')
            plt.imshow(label)#, cmap='gray')
            text = "Label shape "+str(label.shape)+"\n"+self.dynamicRangeInImage(label)
            fig.gca().set(xlabel=text, xticks=[], yticks=[])
            k += 3

        plt.show()
        # also show dimensions, channels, dynamic range of each, occurances in the label (0, 1)

    def viewQuadrupples(self, lefts, rights, labels, predicted, how_many=3, off=0):
        rows, columns = how_many, 4
        fig = plt.figure(figsize=(10, 8))
        k = 1
        for i in range(how_many):
            #idx = i + random.randint(1, len(lefts)-how_many-off)
            idx = i

            left = lefts[idx+off]
            fig.add_subplot(rows, columns, k)
            if left.shape[2] > 3:
                plt.imshow(left[:,:,1:4])
            else:
                plt.imshow(left)
            text = "Left shape "+str(left.shape)+"\n"+self.dynamicRangeInImage(left)[0:-2]
            fig.gca().set(xlabel=text, xticks=[], yticks=[])

            right = rights[idx+off]
            fig.add_subplot(rows, columns, k+1)
            if right.shape[2] > 3:
                plt.imshow(right[:,:,1:4])
            else:
                plt.imshow(right)
            text = "Right shape "+str(right.shape)+"\n"+self.dynamicRangeInImage(right)[0:-2]
            fig.gca().set(xlabel=text, xticks=[], yticks=[])

            label = labels[idx + off]
            fig.add_subplot(rows, columns, k+2)
            #plt.imshow(label, cmap='gray')
            plt.imshow(label, cmap='gray')
            text = "Label shape "+str(label.shape)+"\n"+self.dynamicRangeInImage(label)
            fig.gca().set(xlabel=text, xticks=[], yticks=[])

            one_predicted = predicted[idx+off]
            fig.add_subplot(rows, columns, k+3)
            #plt.imshow(label, cmap='gray')
            plt.imshow(one_predicted)#, cmap='gray')
            text = "Predicted shape "+str(one_predicted.shape)+"\n"+self.dynamicRangeInImage(one_predicted)
            fig.gca().set(xlabel=text, xticks=[], yticks=[])
            k += 4


        plt.show()
        # also show dimensions, channels, dynamic range of each, occurances in the label (0, 1)

    def explore_set_stats(self,arr_set):
        amin = np.amin(arr_set.flatten())
        amax = np.amax(arr_set.flatten())
        print("   min", amin, "max", amax, " ... avg", np.mean(arr_set.flatten()), "+-", np.std(arr_set.flatten()),"   SetShape:",arr_set.shape)

    ### TRAINING VISUALIZATIONS:

    def nice_plot_history(self, history, no_val=False):
        fig, ax = plt.subplots()

        loss = history.history["loss"]
        accuracy = history.history["accuracy"]
        if not no_val:
            val_loss = history.history["val_loss"]
            val_accuracy = history.history["val_accuracy"]

        data = [loss, accuracy]
        columns = ['loss', 'accuracy']

        if not no_val:
            data = [loss, val_loss, accuracy, val_accuracy]
            columns = ['loss', 'val_loss', 'accuracy', 'val_accuracy']

        df = pd.DataFrame(data)
        df = df.transpose()
        df.columns = columns

        print(df)

        accuracies = []
        losses = []

        def plot_item(name, color, max_wanted=True):
            line_item = sns.lineplot(y=name, x=df.index, data=df, label=name)
            if max_wanted:
                max_y = df[name].max()
                max_idx = df[name].idxmax()
            else:
                max_y = df[name].min()
                max_idx = df[name].idxmin()

            text_item = plt.text(max_idx + 0.2, max_y + 0.05, str(round(max_y, 2)), horizontalalignment='left',
                                 size='medium', color=color, weight='semibold')
            return [line_item, text_item]

        accuracies += plot_item("accuracy", "blue")
        if not no_val:
            accuracies += plot_item("val_accuracy", "orange")

        losses += plot_item("loss", "green", max_wanted=False)
        max_val = df["accuracy"].max()

        if not no_val:
            losses += plot_item("val_loss", "brown", max_wanted=False)
            max_val = max(df["loss"].max(), df["val_loss"].max())

        plt.ylim(0, 1)
        plt.ylabel("Accuracy")

        plt.legend(loc='lower right')  # best

        def press(event):
            sys.stdout.flush()
            if event.key == '+':
                # zoom to 0-1 accuracy
                plt.ylim(0, 1)
            elif event.key == '-':
                plt.ylim(0, max_val)
            elif event.key == 'b':
                plt.legend(loc='best')
            elif event.key == 'a':
                plt.legend(loc='lower right')
            else:
                print('press', event.key)

            fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', press)

        plt.show()


    # Data checking:

    def check_balance_of_data(self, labels, optional_paths=''):
        # In this we want to check how many pixels are marking "change" in each image

        #labels = labels[0:500]

        exploration_sum_values = {}
        array_of_number_of_change_pixels = []

        # slow part >>
        for image in tqdm(labels):
            values = self.occurancesInImage(image) # values we have are "1" and "0.0"
            #print("values.keys()",values.keys())
            for value in values:
                #print("'",value,"'")
                if value in exploration_sum_values:
                    exploration_sum_values[value] += values[value]
                else:
                    exploration_sum_values[value] = values[value]
            if 1 in values.keys(): # number 1 as a key signifies changed pixel
                array_of_number_of_change_pixels.append(values[1])
            else:
                array_of_number_of_change_pixels.append(0)

        print("In the whole dataset, we have these values:")
        print(exploration_sum_values)

        print("We have these numbers of alive pixels:")
        print(array_of_number_of_change_pixels)

        self.save_arr(array_of_number_of_change_pixels)

        # << skip it, if you can
        array_of_number_of_change_pixels = self.load_arr()

        array_of_number_of_change_pixels = array_of_number_of_change_pixels / (256*256) * 100.0 # percentage of image changed

        bigger_than_percent = 10.0

        idx_examples_bigger = np.argwhere(array_of_number_of_change_pixels > bigger_than_percent)
        original_array_of_number_of_change_pixels = array_of_number_of_change_pixels

        less = [val for val in array_of_number_of_change_pixels if val <= bigger_than_percent]
        array_of_number_of_change_pixels = [val for val in array_of_number_of_change_pixels if val > bigger_than_percent] # no zeros
        print("The data which is >",bigger_than_percent,"% changed is ", len(array_of_number_of_change_pixels), "versus the remainder of", len(less))

        # the histogram of the data
        #fig = plt.figure(figsize=(10, 8))
        fig = plt.figure()
        bins = 100
        values_of_bins, bins, patches = plt.hist(array_of_number_of_change_pixels, bins, facecolor='g', alpha=0.75)

        print("values_of_bins", np.asarray(values_of_bins).astype(int))
        print("bins sizes", bins)
        plt.yscale('log', nonposy='clip')

        plt.title('How much change in the 256x256 tiles?')
        plt.xlabel('Percentage of pixels belonging to change')
        plt.ylabel('Log scale of number of images/256x256 tiles')

        plt.show()

        if optional_paths is not '':
            labels_to_show = []
            txt_labels = []
            for i in range(100):
                idx = idx_examples_bigger[i][0]
                label_image = optional_paths[idx]
                labels_to_show.append(label_image)
                txt_labels.append(original_array_of_number_of_change_pixels[idx])

            import DataLoader
            images = [DataLoader.DataLoader.load_vector_image(0, path) for path in labels_to_show]

            self.viewVectors(images, txt_labels, how_many=6, off=0)
            self.viewVectors(images, txt_labels, how_many=6, off=6)
            self.viewVectors(images, txt_labels, how_many=6, off=12)

    # File helpers

    def mkdir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_arr(self, arr):

        self.mkdir(self.settings.large_file_folder+"debuggerstuffs")
        hdf5_path = self.settings.large_file_folder+"debuggerstuffs/savedarr.h5"

        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("arr", data=arr, dtype="float32")
        hdf5_file.close()

        print("Saved arr to:", hdf5_path)

    def load_arr(self):
        hdf5_path = self.settings.large_file_folder+"debuggerstuffs/savedarr.h5"

        hdf5_file = h5py.File(hdf5_path, "r")
        arr = hdf5_file['arr'][:]
        hdf5_file.close()
        return arr
