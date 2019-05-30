#parser.add_argument('-name', help='run name - will output in this dir', default="Run-"+month+"-"+day)
#parser.add_argument('-model_epochs', help='How many epochs will each model train?', default="50")
#parser.add_argument('-model_batchsize', help='How big batch size for each model? (This is limited by the GPUs memory)', default="4")
#parser.add_argument('-train_augmentation', help='Turn on augmentation? (one new image for each image in the train set - effectively doubling the dataset size)', default="False")
#parser.add_argument('-AL_iterations', help='Number of iterations in the Active Learning loop', default="10")
#parser.add_argument('-AL_initialsample_size', help='Start with this many sampled images in the training set', default="100")
#parser.add_argument('-AL_testsample_size', help='Have this many balanced sample images in the testing set', default="250")
#parser.add_argument('-AL_iterationsample_size', help='Add this many images in each iteration', default="100")
#parser.add_argument('-AL_method', help='Sampling method (choose from "Random", "Ensemble")', default="Random")
#parser.add_argument('-AL_Ensemble_numofmodels', help='If we chose Ensemble, how many models are there?', default="3")


python3 ActiveLearningLoop.py -AL_method Random -name TryingRandomOnFull

python3 ActiveLearningLoop.py -AL_method Ensemble -name TryingEnsembleOnFull

#python3 ActiveLearningLoop.py -AL_method Ensemble -AL_Ensemble_numofmodels 5 -name TryingEnsembleWith5models
