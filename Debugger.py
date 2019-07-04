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

    def inspect_dataset(self,data,paths, to_check = 12):
        print("We have this data: L", data[0].shape, "R",data[1].shape, "V", data[2].shape)
        print("We have this paths: L", len(paths[0]), "R",len(paths[1]), "V", len(paths[2]))
        N_data = data[0].shape[0]
        N_paths = len(paths[0])

        to_check = np.min([to_check,N_data])
        indices = random.sample(range(0, N_data), to_check)
        lefts = []
        rights = []
        labels = []
        txts = []

        for idx in indices:
            lefts.append(data[0][idx])
            rights.append(data[1][idx])
            labels.append(data[2][idx])
            if N_data == N_paths:
                txts.append(paths[0][idx][-20:] + "/" + paths[1][idx][-20:] + "/")
                if paths[2][idx] is not None:
                    txts[-1] += paths[2][idx][-20:]
                else:
                    txts[-1] += "None"
                txts[-1] += "\n"
            #print(idx," : ", txts[-1])

        checked = 0
        while checked < to_check:
            self.viewTripples(lefts, rights, labels, txts, how_many=3, off=checked)
            checked += 3

    def check_paths(self, left_paths, right_paths, label_paths, to_check = 12):
        N_paths = len(left_paths)
        to_check = np.min([to_check,N_paths])
        indices = random.sample(range(0, N_paths), to_check)

        for idx in indices:
            l = left_paths[idx].split("/")[-1]
            r = right_paths[idx].split("/")[-1]
            if label_paths[idx] is not None:
                v = label_paths[idx].split("/")[-1]
            else:
                v = "None"
            print(l, ",\t\t", r , ",\t\t", v)


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
    def viewVectors(self, images, texts=[], how_many=6, off=0):

        rows, columns = 2, 3
        #fig = plt.figure(figsize=(10, 8))
        fig = plt.figure()
        k = 1
        for idx in range(how_many):

            label = images[idx+off]
            fig.add_subplot(rows, columns, k)
            plt.imshow(label, cmap='gray')

            text = ""
            if len(texts)>0:
                text += str(round(texts[idx + off], 2)) + "%"
            #text += "Label shape "+str(label.shape)+"\n"+self.dynamicRangeInImage(label)
            #text = ""
            fig.gca().set(xlabel=text, xticks=[], yticks=[])
            k += 1

        plt.show()
        # also show dimensions, channels, dynamic range of each, occurances in the label (0, 1)

    def viewTrippleFromUrl(self, left_path, right_path, label_path, optional_title = ""):
        fig = plt.figure(figsize=(10, 8))
        from skimage import io
        IMAGE_RESOLUTION = 112

        def load_vector_image(filename):
            if filename == None:
                arr = np.zeros((IMAGE_RESOLUTION, IMAGE_RESOLUTION), dtype=float)
                return arr
            img = io.imread(filename)
            arr = np.asarray(img)

            print("occurances before thr:",self.occurancesInImage(arr))

            ## FOR NEWER DATASETS
            arr[arr == 0] = 0
            arr[arr == 65535] = 0
            arr[arr != 0] = 1

            print("occurances after thr:",self.occurancesInImage(arr))

            ## FOR OLDER DATASETS
            #thr = 0
            #arr[arr > thr] = 1
            #arr[arr <= thr] = 0
            return arr

        def load_raster_image(filename):
            img = io.imread(filename)
            arr = np.asarray(img)
            return arr

        left = load_raster_image(left_path)
        fig.add_subplot(1, 3, 1)
        if left.shape[2] > 3:
            plt.imshow(left[:, :, 1:4])
        else:
            plt.imshow(left)
        text = "Left shape " + str(left.shape) + "\n" + self.dynamicRangeInImage(left)[0:-2]
        fig.gca().set(xlabel=text, xticks=[], yticks=[])

        right = load_raster_image(right_path)
        fig.add_subplot(1, 3, 2)
        if right.shape[2] > 3:
            plt.imshow(right[:, :, 1:4])
        else:
            plt.imshow(right)

        text = "Right shape " + str(right.shape) + "\n" + self.dynamicRangeInImage(right)[0:-2]
        fig.gca().set(xlabel=text, xticks=[], yticks=[])

        label = load_vector_image(label_path)
        fig.add_subplot(1, 3, 3)
        plt.imshow(label)  # , cmap='gray')

        text = ""
        text += "Label shape " + str(label.shape) + "\n" + self.dynamicRangeInImage(label)
        fig.gca().set(xlabel=text, xticks=[], yticks=[])

        if len(optional_title) > 0:
            plt.title(optional_title)

        plt.show()
        # also show dimensions, channels, dynamic range of each, occurances in the label (0, 1)

    def viewTripples(self, lefts, rights, labels, txts=[], how_many=3, off=0):
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

            text = ""
            if len(txts) > 0:
                text += txts[idx+off]
            text += "Label shape "+str(label.shape)+"\n"+self.dynamicRangeInImage(label)
            fig.gca().set(xlabel=text, xticks=[], yticks=[])
            k += 3

        plt.show()
        # also show dimensions, channels, dynamic range of each, occurances in the label (0, 1)

    def viewQuadrupples(self, lefts, rights, labels, predicted, txts=[], how_many=3, off=0, show=True, save=False, name="lastplot", show_txts = True):
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
            if show_txts:
                fig.gca().set(xlabel=text, xticks=[], yticks=[])
            else:
                fig.gca().set(xlabel="", xticks=[], yticks=[])

            right = rights[idx+off]
            fig.add_subplot(rows, columns, k+1)
            if right.shape[2] > 3:
                plt.imshow(right[:,:,1:4])
            else:
                plt.imshow(right)
            text = "Right shape "+str(right.shape)+"\n"+self.dynamicRangeInImage(right)[0:-2]
            if show_txts:
                fig.gca().set(xlabel=text, xticks=[], yticks=[])
            else:
                fig.gca().set(xlabel="", xticks=[], yticks=[])

            label = labels[idx + off]
            fig.add_subplot(rows, columns, k+2)
            #plt.imshow(label, cmap='gray')
            plt.imshow(label, cmap='gray')
            text = "Label shape "+str(label.shape)+"\n"+self.dynamicRangeInImage(label)
            if show_txts:
                fig.gca().set(xlabel=text, xticks=[], yticks=[])
            else:
                fig.gca().set(xlabel="", xticks=[], yticks=[])

            one_predicted = predicted[idx+off]
            fig.add_subplot(rows, columns, k+3)
            #plt.imshow(label, cmap='gray')
            plt.imshow(one_predicted)#, cmap='gray')
            text = ""
            if len(txts) > 0:
                text += txts[idx+off]
            text += "Predicted shape "+str(one_predicted.shape)+"\n"+self.dynamicRangeInImage(one_predicted)
            if show_txts:
                fig.gca().set(xlabel=text, xticks=[], yticks=[])
            else:
                fig.gca().set(xlabel="", xticks=[], yticks=[])
            k += 4

        if show:
            plt.show()
        if save:
            plt.savefig(name+".png")
        # also show dimensions, channels, dynamic range of each, occurances in the label (0, 1)

    def explore_set_stats(self,arr_set):
        amin = np.amin(arr_set.flatten())
        amax = np.amax(arr_set.flatten())
        print("   min", amin, "max", amax, " ... avg", np.mean(arr_set.flatten()), "+-", np.std(arr_set.flatten()),"   SetShape:",arr_set.shape)

    ### TRAINING VISUALIZATIONS:

    def nice_plot_history(self, history, added_plots = [], no_val=False, show=True, save=False, name="lastplot", max_y = 1.0):
        fig, ax = plt.subplots()

        loss = history.history["loss"]
        accuracy = history.history["acc"]
        if not no_val:
            val_loss = history.history["val_loss"]
            val_accuracy = history.history["val_acc"]

        data = [loss, accuracy]
        columns = ['loss', 'accuracy']

        if not no_val:
            data = [loss, val_loss, accuracy, val_accuracy]
            columns = ['loss', 'val_loss', 'acc', 'val_acc']


        if len(added_plots)>0:
            for item in added_plots:
                data += [history.history[item]]
                columns += [item]

        if self.settings.verbose > 2:
            print(data)
            print(columns)

        df = pd.DataFrame(data)
        df = df.transpose()
        df.columns = columns

        if self.settings.verbose > 2:
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

        accuracies += plot_item("acc", "blue")
        if not no_val:
            accuracies += plot_item("val_acc", "orange")

        losses += plot_item("loss", "green", max_wanted=False)
        max_val = df["acc"].max()

        if not no_val:
            losses += plot_item("val_loss", "brown", max_wanted=False)
            max_val = max(df["loss"].max(), df["val_loss"].max())

        if len(added_plots)>0:
            for item in added_plots:
                plot_item(item, "grey")

        plt.ylim(0, max_y)
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

        if show:
            plt.show()
        if save:
            plt.savefig(name+".png")
            #plt.savefig(name+".pdf")

        plt.close()


    # File helpers

    def mkdir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_arr(self, arr, specialname = ""):

        suceeded = False

        while not suceeded:
            try:

                self.mkdir(self.settings.large_file_folder+"debuggerstuffs")
                hdf5_path = self.settings.large_file_folder+"debuggerstuffs/savedarr"+specialname+".h5"

                hdf5_file = h5py.File(hdf5_path, mode='w')
                hdf5_file.create_dataset("arr", data=arr, dtype="float32")
                hdf5_file.close()

                print("Saved arr to:", hdf5_path)

                suceeded = True

            except Exception as e:
                print("exception, retrying e=",e)

                suceeded = False

    def load_arr(self, specialname = ""):
        suceeded = False

        while not suceeded:
            try:

                hdf5_path = self.settings.large_file_folder+"debuggerstuffs/savedarr"+specialname+".h5"

                hdf5_file = h5py.File(hdf5_path, "r")
                arr = hdf5_file['arr'][:]
                hdf5_file.close()

                suceeded = True

            except Exception as e:
                print("exception, retrying e=",e)

                suceeded = False

        return arr
