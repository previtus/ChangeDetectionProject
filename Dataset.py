import DataLoader, DataPreprocesser

class Dataset(object):
    """
    Will handle the dataset
    """

    def __init__(self, settings):
        self.settings = settings
        self.dataLoader = DataLoader.DataLoader(settings)
        self.dataPreprocesser = DataPreprocesser.DataPreprocesser(settings)

        self.dataset = self.dataLoader.dataset
        print("Dataset loaded with", len(self.dataset[0]), "images.")

        # preprocess the dataset
        self.dataset = self.dataPreprocesser.process_dataset(self.dataset)