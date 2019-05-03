# === Initialize sets - Unlabeled, Train and Test

from ActiveLearning.LargeDatasetHandler_AL import tmp_get_whole_dataset
from ActiveLearning.LargeDatasetHandler_AL import LargeDatasetHandler_AL
from ActiveLearning.ModelHandler_dataIndependent import ModelHandler_dataIndependent
from ActiveLearning.DataPreprocesser_dataIndependent import DataPreprocesser_dataIndependent
from ActiveLearning.TrainTestHandler import TrainTestHandler
from Evaluator import Evaluator
from timeit import default_timer as timer

#in_memory = False
in_memory = True
RemainingUnlabeledSet = tmp_get_whole_dataset(in_memory)
settings = RemainingUnlabeledSet.settings

TestSet = LargeDatasetHandler_AL(settings, "inmemory")
selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(250)
packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
TestSet.add_items(packed_items)
print("Test set:")
TestSet.report()

TrainSet = LargeDatasetHandler_AL(settings, "inmemory")

# === Model and preprocessors


test_data, test_paths = TestSet.get_all_data_as_arrays()

dataPreprocesser = DataPreprocesser_dataIndependent(settings, number_of_channels=4)
trainTestHandler = TrainTestHandler(settings)
evaluator = Evaluator(settings)

# === Start looping:
recalls = []
precisions = []
accuracies = []
f1s = []
xs_number_of_data = []

for active_learning_iteration in range(3): #18 also mem problems!
    print("\n")
    print("==========================================================")
    print("\n")

    print("Active Learning iteration:", active_learning_iteration)
    start = timer()

    selected_indices = RemainingUnlabeledSet.sample_random_indice_subset(100)
    print(selected_indices)
    packed_items = RemainingUnlabeledSet.pop_items(selected_indices)
    TrainSet.add_items(packed_items)

    print("Unlabeled:")
    RemainingUnlabeledSet.report()
    print("Train:")
    TrainSet.report()

    train_data, train_paths = TrainSet.get_all_data_as_arrays()
    print("fitting the preprocessor on this dataset")

    dataPreprocesser.fit_on_train(train_data)

    # non destructive copies of the sets
    processed_train = dataPreprocesser.apply_on_a_set_nondestructively(train_data)
    processed_test = dataPreprocesser.apply_on_a_set_nondestructively(test_data)

    print("Train shapes:", processed_train[0].shape, processed_train[1].shape, processed_train[2].shape)
    print("Test shapes:", processed_test[0].shape, processed_test[1].shape, processed_test[2].shape)

    # Init model
    ModelHandler = ModelHandler_dataIndependent(settings, BACKBONE='resnet34')
    model = ModelHandler.model

    # === Train!:
    print("Now I would train ...")
    epochs = 10 + active_learning_iteration # hmmmmmmm regularize?
    epochs = 1
    batch = 16 # batch 16 for resnet50
    batch = 32
    trainTestHandler.train(model, processed_train, epochs, batch)

    print("Now I would test ...")

    DEBUG_SAVE_ALL_THR_PLOTS = "iteration_"+str(active_learning_iteration).zfill(2)+"_debugThrOverview"
    #DEBUG_SAVE_ALL_THR_PLOTS = None
    recall, precision, accuracy, f1 = trainTestHandler.test(model, processed_test, evaluator, postprocessor=dataPreprocesser, DEBUG_SAVE_ALL_THR_PLOTS=DEBUG_SAVE_ALL_THR_PLOTS)
    recalls.append(recall)
    precisions.append(precision)
    accuracies.append(accuracy)
    f1s.append(f1)
    xs_number_of_data.append(TrainSet.get_number_of_samples())

    print("Now I would store/save the resulting model ...")

    del model
    del train_data
    del processed_train
    del processed_test

    end = timer()
    time = (end - start)
    print("This iteration took "+str(time)+"s ("+str(time/60.0)+"min)")



#- AL outline, have two sets - TrainSet and UnlabeledSet and move data in between them... (always everything we have?)
#2b.) Acquisition function to select subset of the RemainingUnlabeledSet -> move it to the TrainSet

print("recalls:", recalls)
print("precisions:", precisions)
print("accuracies:", accuracies)
print("f1s:", f1s)

import matplotlib.pyplot as plt
plt.plot(xs_number_of_data, recalls, 'r', label="recalls")
plt.plot(xs_number_of_data, precisions, 'b', label="precisions")
plt.plot(xs_number_of_data, accuracies, 'g', label="accuracies")
plt.plot(xs_number_of_data, f1s, 'orange', label="f1s")
plt.legend()
plt.ylim(0.0, 1.0)

plt.show()

"""
eeeee

for e in range(3):
    print("epoch %d" % e)
    # mode > indices, dataonly, datalabels
    for batch in TrainSet.generator_for_all_images(500, mode='datalabels'): # Yields a large batch sample
        indices = batch[0]
        print("indices from", indices[0], "to", indices[-1])
"""