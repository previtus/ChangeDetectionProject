import matplotlib.pyplot as plt
import numpy as np

class Debugger(object):
    """
    Will have functions useful for debugging.
    """



    def __init__(self):
        #self.dataset = dataset
        a = 0

    #def dynamicRangeInSet(self, set_of_images):
    #    return 0

    def dynamicRangeInImage(self, image):
        ranges = ""
        if len(image.shape) > 2:
            for channel in range(image.shape[2]):
                min_val = np.min(image[:,:,channel])
                max_val = np.max(image[:,:,channel])
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


    def viewTripples(self, lefts, rights, labels, how_many=3, off=0):
        #for i in range(len(lefts)):
        #    print(i, "=>", lefts[i].shape, rights[i].shape, labels[i].shape)

        rows, columns = how_many, 3
        fig = plt.figure(figsize=(10, 8))
        k = 1
        for i in range(how_many):
            left = lefts[i+off]
            fig.add_subplot(rows, columns, k)
            plt.imshow(left[:,:,1:4])
            text = "Left shape "+str(left.shape)+"\n"+self.dynamicRangeInImage(left)[0:-2]
            fig.gca().set(xlabel=text, xticks=[], yticks=[])

            right = rights[i+off]
            fig.add_subplot(rows, columns, k+1)
            plt.imshow(right[:,:,1:4])
            text = "Right shape "+str(right.shape)+"\n"+self.dynamicRangeInImage(right)[0:-2]
            fig.gca().set(xlabel=text, xticks=[], yticks=[])

            label = labels[i+off]
            fig.add_subplot(rows, columns, k+2)
            plt.imshow(label, cmap='gray')
            text = "Label shape "+str(label.shape)+"\n"+self.dynamicRangeInImage(label)
            fig.gca().set(xlabel=text, xticks=[], yticks=[])
            k += 3

        plt.show()
        # also show dimensions, channels, dynamic range of each, occurances in the label (0, 1)