import DataLoader, DataPreprocesser, Debugger
import DatasetInstance_OurAerial, DatasetInstance_ONERA

class Dataset(object):
    """
    Will handle the dataset
    """

    def __init__(self, settings):
        self.settings = settings
        self.dataLoader = DataLoader.DataLoader(settings)
        self.debugger = Debugger.Debugger(settings)

        Using_Model1b_Needing_Labels = False

        # just "_clean" is rubbis!
        # manual cleanup at 256_clean2 - 256x256 without overlap
        #dataset_variant = "256"
        dataset_variant = "256_cleanManual"
        ###dataset_variant = "6368_special"
        #dataset_variant = "256"
        #dataset_variant = "112_clean"
        self.datasetInstance = DatasetInstance_OurAerial.DatasetInstance_OurAerial(settings, self.dataLoader, dataset_variant)
        #self.datasetInstance = DatasetInstance_ONERA.DatasetInstance_ONERA(settings, self)

        self.dataPreprocesser = DataPreprocesser.DataPreprocesser(settings,self.datasetInstance)

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

    def shuffle_thyself(self, data):
        # !
        return data

