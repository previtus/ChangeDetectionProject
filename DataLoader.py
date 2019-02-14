from PIL import Image, ImageDraw
import numpy as np

class DataLoader(object):
    """
    Will handle loading and parsing the data.
    """


    def __init__(self, settings):
        self.settings = settings

