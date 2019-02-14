from PIL import Image, ImageDraw
import numpy as np

class DataPreprocesser(object):
    """
    Will handle image editing.
    """


    def __init__(self, settings):
        self.settings = settings
