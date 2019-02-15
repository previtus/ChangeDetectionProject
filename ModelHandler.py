import DataLoader, DataPreprocesser, Dataset, Debugger, Settings

class ModelHandler(object):
    """
    An intermediate between the code and bunch of tester models.
    """

    def __init__(self, settings):
        self.settings = settings
