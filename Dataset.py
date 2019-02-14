import DataLoader, DataPreprocesser, Dataset, Debugger, Settings

class Dataset(object):
    """
    Will handle the dataset
    """



    def __init__(self, settings):
        self.settings = settings
        self.dataLoader = DataLoader.DataLoader(settings)
        self.dataPreprocesser = DataPreprocesser.DataPreprocesser(settings)

