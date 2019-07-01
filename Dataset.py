import DataLoader, DataPreprocesser, Debugger
import DatasetInstance_OurAerial
import numpy as np


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
        dataset_variant = "256_cleanManual"
        ###dataset_variant = "6368_special"
        self.datasetInstance = DatasetInstance_OurAerial.DatasetInstance_OurAerial(self.settings, self.dataLoader, dataset_variant)

        number_of_channels = self.datasetInstance.CHANNEL_NUMBER
        self.dataPreprocesser = DataPreprocesser.DataPreprocesser(self.settings,number_of_channels)

        self.data, self.paths = self.datasetInstance.load_dataset()

        if self.settings.verbose >= 3:
            self.debugger.inspect_dataset(self.data, self.paths, 3) # 3

        print("Dataset loaded with", len(self.data[0]), "images.")

        # Split into training, validation and test:

        K = self.settings.TestDataset_K_Folds
        test_fold = self.settings.TestDataset_Fold_Index
        print("K-Fold crossval: [",test_fold,"from",K,"]")
        self.train, self.val, self.test = self.datasetInstance.split_train_val_test_KFOLDCROSSVAL(self.data, test_fold=test_fold, K=K)
        self.paths = np.asarray(self.paths)
        self.train_paths, self.val_paths, self.test_paths = self.datasetInstance.split_train_val_test_KFOLDCROSSVAL(self.paths, test_fold=test_fold, K=K)

        print("Has ", len(self.train[0]), "train, ", len(self.val[0]), "val, ", len(self.test[0]), "test, ")
        print("Has ", len(self.train_paths[0]), "train_paths, ", len(self.val_paths[0]), "val_paths, ", len(self.test_paths[0]), "test_paths, ")

        #print("Revert...")
        #self.train, self.val, self.test = self.datasetInstance.split_train_val_test(self.data)
        #self.train_paths, self.val_paths, self.test_paths = self.datasetInstance.split_train_val_test(self.paths)
        #print("Has ", len(self.train[0]), "train, ", len(self.val[0]), "val, ", len(self.test[0]), "test, ")

        # preprocess the dataset
        self.train, self.val, self.test = self.dataPreprocesser.process_dataset(self.train, self.val, self.test)

        #same_test_val_check = np.array_equal(self.val, self.test)
        #print("Is the test the same as val? ",same_test_val_check )