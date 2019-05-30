#parser = argparse.ArgumentParser(description='Project: Change detection on aerial images.')
#parser.add_argument('-name', help='run name - will output in this dir', default="Run-"+month+"-"+day)
#parser.add_argument('-train_epochs', help='How many epochs', default='100')
#parser.add_argument('-train_batch', help='How big batch size', default='8')
#   path = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/"
#   star = '*resnet101-8batch_Augmentation1to1_ClassWeights1to3_[KFold_*'
#   model_used = "resnet101"
#   star = '*resnet50-8batch_Augmentation1to1_ClassWeights1to3_[KFold_*'
#   model_used = "resnet50"
#parser.add_argument('-model_backend', help='Model used in the encoder part of the U-Net structures model', default=model_used)
#parser.add_argument('-models_path_star', help='Path to models with reg exp selection', default=path+star)
#parser.add_argument('-input_file', help='Alternatively we can have a file with specified input models and their specific settings.', default=INPUT_FILE_EXCLUSIONS)

#INPUT_FILE_EXCLUSIONS = "/home/ruzickav/python_projects/ChangeDetectionProject/__OUTPUTS/ResNet101_ManualExclusions.txt"
#INPUT_FILE_EXCLUSIONS = "/home/ruzickav/python_projects/ChangeDetectionProject/__OUTPUTS/ResNet50_ManualExclusions.txt"

# ~ depending on what's set in main_evaluate, this might be loading additional dataset for checking

cd /home/ruzickav/python_projects/ChangeDetectionProject/
python3 main_evaluate.py -model_backend resnet50 -models_path_star "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/*resnet50-8batch_Augmentation1to1_ClassWeights1to3_[KFold_*" -input_file "/home/ruzickav/python_projects/ChangeDetectionProject/__OUTPUTS/ResNet50_ManualExclusions.txt"
python3 main_evaluate.py -model_backend resnet101 -models_path_star "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/*resnet101-8batch_Augmentation1to1_ClassWeights1to3_[KFold_*" -input_file "/home/ruzickav/python_projects/ChangeDetectionProject/__OUTPUTS/ResNet101_ManualExclusions.txt"
#cd /scratch/ruzicka/python_projects_large/progressive_growing_of_gans
#python3 train.py
