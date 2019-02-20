import DataLoader, DataPreprocesser, Debugger
import DatasetInstance_OurAerial, DatasetInstance_ONERA

class Dataset(object):
    """
    Will handle the dataset
    """

    def __init__(self, settings):
        self.settings = settings
        self.dataLoader = DataLoader.DataLoader(settings)
        self.dataPreprocesser = DataPreprocesser.DataPreprocesser(settings)
        self.debugger = Debugger.Debugger(settings)

        dataset_variant = 112
        self.datasetInstance = DatasetInstance_OurAerial.DatasetInstance_OurAerial(settings, self.dataLoader, dataset_variant)
        #self.datasetInstance = DatasetInstance_ONERA.DatasetInstance_ONERA(settings, self)

        self.data, self.paths = self.datasetInstance.load_dataset()
        self.debugger.inspect_dataset(self.data, self.paths)

        print("Dataset loaded with", len(self.data[0]), "images.")

        # preprocess the dataset
        self.data = self.dataPreprocesser.process_dataset(self.data)