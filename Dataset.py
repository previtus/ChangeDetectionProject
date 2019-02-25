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

        if self.settings.verbose >= 3:
            self.debugger.inspect_dataset(self.data, self.paths, 3) # 3

        print("Dataset loaded with", len(self.data[0]), "images.")

        # Shuffle
        self.data = self.shuffle_thyself(self.data)

        # Split into training, validation and test:
        self.train, self.val, self.test = self.datasetInstance.split_train_val_test(self.data)
        print("Has ", len(self.train[0]), "train, ", len(self.val[0]), "val, ", len(self.test[0]), "test, ")

        # preprocess the dataset


        self.original_test = self.test
        self.train, self.val, self.test = self.dataPreprocesser.process_dataset_Train(self.train, self.val, self.test)


        #self.train = self.dataPreprocesser.process_dataset(self.train)
        #self.val = self.dataPreprocesser.process_dataset(self.val)
        #self.test = self.dataPreprocesser.process_dataset(self.test)

        # left images (train)
        #    min -0.5 max 0.5  ... avg -0.08794274622727885 +- 0.17829913314185908    SetShape: (2200, 112, 112, 3)
        # right images (train)
        #    min -0.5 max 0.5  ... avg -0.044110019007602995 +- 0.19196525731903477    SetShape: (2200, 112, 112, 3)

    def shuffle_thyself(self, data):
        # !
        return data