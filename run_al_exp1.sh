#!/bin/sh


# identification
# parser.add_argument('-name', help='run name - will output in this dir', default="Run-"+month+"-"+day)
# parser.add_argument('-seed', help='random seed (for multiple runs)', default="30")
# parser.add_argument('-uid', help='Unique ID (for automatic runners to differentiate between files)', default="O")

# interesting ones:
# parser.add_argument('-AL_method', help='Sampling method (choose from "Random", "Ensemble", "MonteCarloBatchNormalization")', default="Ensemble")
# parser.add_argument('-AL_AcquisitionFunction', help='For any method other than Random (choose from "Variance", "Entropy", "BALD")', default="Variance")
# parser.add_argument('-AL_Ensemble_numofmodels', help='If we chose Ensemble, how many models are there?', default="5")
# parser.add_argument('-AL_MCBN_numofruns', help='If we chose Ensemble, how many models are there?', default="5")

# parser.add_argument('-DEBUG_remove_from_dataset', help='Debug to remove random samples without change from the original dataset...', default="40000")

# parser.add_argument('-model_batchsize', help='How big batch size for each model? (This is limited by the GPUs memory)', default="16")

# parser.add_argument('-AL_iterations', help='Number of iterations in the Active Learning loop', default="10")

# parser.add_argument('-AL_initialsample_size', help='Start with this many sampled images in the training set', default="50")
# parser.add_argument('-AL_iterationsample_size', help='Add this many images in each iteration', default="100")

# others:
#parser.add_argument('-model_epochs', help='How many epochs will each model train?', default="100") #50? 100?
#parser.add_argument('-model_backbone', help='Encoder', default="resnet34")
#parser.add_argument('-train_augmentation', help='Turn on augmentation? (one new image for each image in the train set - effectively doubling the dataset size)', default="False")
#parser.add_argument('-AL_testsample_size', help='Have this many balanced sample images in the testing set (used for plots)', default="200")
#parser.add_argument('-AL_valsample_size', help='Have this many balanced sample images in the validation set (used for automatic thr choice and val errs)', default="200")
#parser.add_argument('-DEBUG_loadLastALModels', help='Debug function - load last saved model weights instead of training ...', default="False")

# bsub -n 1 -W 120:00 -R "rusage[mem=64000, ngpus_excl_p=1]" python3 XYZ
# python3 ActiveLearningLoop.py XYZ


#for seed in 10 20 30 40 50 60 70 80 90 100


# Experiment 1 - run five MCBN with batch size = 4

uid="runMCBNBatchSize4FiveTimes"
for seed in 10 20 30 40 50
do
    name="CompRuns3_"$seed"_"
    echo bsub -n 1 -W 120:00 -R "rusage[mem=64000, ngpus_excl_p=1]" python3 ActiveLearningLoop.py -name $name -uid $uid"_"$seed -seed $seed -AL_method MonteCarloBatchNormalization -model_batchsize 4
done