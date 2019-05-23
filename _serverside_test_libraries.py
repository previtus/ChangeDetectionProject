import matplotlib, os
#if not('DISPLAY' in os.environ):
matplotlib.use("Agg")

import argparse
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

fig, ax = plt.subplots()


import DataLoader, DataPreprocesser, Dataset, Debugger, Settings, ModelHandler, Evaluator
from timeit import default_timer as timer
from datetime import *

months = ["unk","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import pandas as pd
import sys
import random
import os
from tqdm import tqdm
import h5py

