# Deep Active Learning in Remote Sensing for data efficient Change Detection

Implementation of the Siamese U-Net model with the pre-trained ResNet34 architecture as an encoder for the Change Detection task on Remote Sensing dataset with support for Deep Active Learning.

<p align="center">
<img src="https://raw.githubusercontent.com/previtus/ChangeDetectionProject/master/_illustration.jpg" width="560">
</p>

## Colab demo: <a href="https://colab.research.google.com/github/previtus/ChangeDetectionProject/blob/master/demo/_ChangeDetection_prediction_example.ipynb" title="Open In Colab"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Check out the basic change detection demo at: <a href="https://github.com/previtus/ChangeDetectionProject/blob/master/demo/_ChangeDetection_prediction_example.ipynb">ChangeDetection_prediction_example.ipynb</a>

## Setup:

Start with installing the prerequisite python libraries. We worked with the following versions: 

```
tensorflow              1.12.0
Keras                   2.2.4
Keras-Applications      1.0.7
Keras-Preprocessing     1.0.5
numpy                   1.16.0
opencv-python-headless  4.0.0.21
scikit-image            0.14.2
scikit-learn            0.20.2
albumentations         0.2.0
image-classifiers      0.2.0
imageio                2.5.0
imageio-ffmpeg         0.2.0
seaborn                0.9.0
segmentation-models    0.2.0
tqdm                   4.29.1
```

Download the dataset and place it into a folder specified in Settings.py. 

To **train a model on the task of change detection** see the "main.py" and run it with the required arguments (such as encoder type, number of epochs or the batch size).

Run this to see the help:
```
python3 main.py --help
```

To **use the deep active learning** algorithms see "ActiveLearningLoop.py".

Run this to see the help:
```
python3 ActiveLearningLoop.py --help
```

These are the example calls for the three tested methods:

```
python3 ActiveLearningLoop.py -name Run1_Ensemble_N5 -AL_method Ensemble -AL_Ensemble_numofmodels 5 -train_augmentation True
python3 ActiveLearningLoop.py -name Run2_MCBN_M5 -AL_method MonteCarloBatchNormalization -AL_MCBN_numofruns 5 -train_augmentation True
python3 ActiveLearningLoop.py -name Run0_Random -AL_method Random -train_augmentation True

# Note we can also use:
# -AL_AcquisitionFunction (choose from "Variance", "Entropy")

# Further experimentation:
# Adding N - this one would add 32 samples for 40 iterations => 1280 samples in the final iteration
python3 ActiveLearningLoop.py -name Run3_Ensemble_N5_Add32 -AL_method Ensemble -AL_Ensemble_numofmodels 5 -AL_iterations 40 -AL_iterationsample_size 32 -train_augmentation True
```
