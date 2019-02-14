import DataLoader, DataPreprocesser, Dataset, Debugger, Settings
from timeit import default_timer as timer
from datetime import *

months = ["unk","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import argparse

parser = argparse.ArgumentParser(description='Project: Change detection on aerial images.')

def main(args):
    print(args)

    settings = Settings.Settings(args)
    dataset = Dataset.Dataset(settings)



if __name__ == '__main__':
    args = parser.parse_args()

    start = timer()

    args.name = "Run"+day+"-"+month

    main(args)

    end = timer()
    time = (end - start)
    #print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")

