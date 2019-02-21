from PIL import Image, ImageDraw
import numpy as np

class DataPreprocesser(object):
    """
    Will handle image editing.
    """


    def __init__(self, settings):
        self.settings = settings

    # to do:
    # channel wise normalization
    # - on training dataset
    # - then use the same values on the val dataset

    def process_dataset(self, dataset):
        lefts, rights, labels = dataset
        # from 0-255 : into 0.0 - 1.0
        lefts = (lefts / 255.0) - 0.5
        rights = (rights / 255.0) - 0.5

        # keep at 0-1 for the sigmoid
        #labels = labels - 0.5
        #labels = labels / 2.0

        return [lefts, rights, labels]

    def postprocess_labels(self, labels):
        # serves to project final labels back to where they originally were

        #labels = (labels + 0.5)
        #labels = labels * 2.0

        return labels

    def postprocess_images(self, images):
        # from -0.5-0.5 back to 0.0-1.0

        images = (images + 0.5)
        # matlibplot at needs positive values, 0.0 -- 1.0

        return images