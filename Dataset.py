import DataLoader, DataPreprocesser, Debugger
import DatasetInstance_OurAerial, DatasetInstance_ONERA

class Dataset(object):
    """
    Will handle the dataset
    """

    def __init__(self, settings, init_source = 1):
        self.settings = settings
        self.dataLoader = DataLoader.DataLoader(settings)
        self.debugger = Debugger.Debugger(settings)

        if init_source == 1:
            self.init_from_stable_datasets()
        else:
            print("Init manually from data and labels")
            self.datasetInstance = None
            self.dataPreprocesser = None

    def init_from_stable_datasets(self):
        Using_Model1b_Needing_Labels = False

        # just "_clean" is rubbis!
        # manual cleanup at 256_clean2 - 256x256 without overlap
        #dataset_variant = "256"
        dataset_variant = "256_cleanManual"
        ###dataset_variant = "6368_special"
        #dataset_variant = "256"
        #dataset_variant = "112_clean"
        self.datasetInstance = DatasetInstance_OurAerial.DatasetInstance_OurAerial(self.settings, self.dataLoader, dataset_variant)
        #self.datasetInstance = DatasetInstance_ONERA.DatasetInstance_ONERA(settings, self)

        number_of_channels = self.datasetInstance.CHANNEL_NUMBER
        self.dataPreprocesser = DataPreprocesser.DataPreprocesser(self.settings,number_of_channels)

        self.data, self.paths = self.datasetInstance.load_dataset()

        if self.settings.verbose >= 3:
            self.debugger.inspect_dataset(self.data, self.paths, 3) # 3

        print("Dataset loaded with", len(self.data[0]), "images.")

        # Shuffle
        self.data = self.shuffle_thyself(self.data)

        # Split into training, validation and test:
        self.train, self.val, self.test = self.datasetInstance.split_train_val_test(self.data)
        self.train_paths, self.val_paths, self.test_paths = self.datasetInstance.split_train_val_test(self.paths)
        print("Has ", len(self.train[0]), "train, ", len(self.val[0]), "val, ", len(self.test[0]), "test, ")

        # preprocess the dataset
        self.train, self.val, self.test = self.dataPreprocesser.process_dataset(self.train, self.val, self.test)

        # only if we use model 1b and if we have our data and not onera
        if Using_Model1b_Needing_Labels:
            self.train_classlabels = self.datasetInstance.mask_label_into_class_label(self.train[2])
            self.val_classlabels = self.datasetInstance.mask_label_into_class_label(self.val[2])
            self.test_classlabels = self.datasetInstance.mask_label_into_class_label(self.test[2])

        #print("Class labels:")
        #print(self.train_classlabels[0:10])
        #print(self.train[2][1][0][0:50])

        #self.train = self.dataPreprocesser.process_dataset_OLDSIMPLE(self.train)
        #self.val = self.dataPreprocesser.process_dataset_OLDSIMPLE(self.val)
        #self.test = self.dataPreprocesser.process_dataset_OLDSIMPLE(self.test)


    def init_from_data_manually(self, train_data, train_paths, test_data, test_paths):
        # all of these come as ~ [lefts, rights, labels] where lefts etc. are arrays
        print("Dataset loaded with", len(train_data[0]), "images.")

        # Shuffle
        train_data = self.shuffle_thyself(train_data)

        # Split into training, validation and test:
        self.train = train_data
        self.val = []
        self.test = test_data

        self.train_paths = train_paths
        self.val_paths = []
        self.test_paths = test_paths

        # preprocess the dataset
        self.train, self.val, self.test = self.dataPreprocesser.process_dataset(self.train, self.val, self.test)

    def shuffle_thyself(self, data):
        # !
        return data

