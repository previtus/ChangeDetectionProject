from PIL import Image, ImageDraw
import numpy as np

class DataPreprocesser(object):
    """
    Will handle image editing.
    """


    def __init__(self, settings):
        self.settings = settings

    def process_dataset(self, dataset):
        lefts, rights, labels = dataset
        # into 0.0 - 1.0
        lefts = lefts / 255.0
        rights = rights / 255.0

        # into 0.0 - 0.5 (output seems to prefer that?)
        labels = labels / 2.0

        return [lefts, rights, labels]