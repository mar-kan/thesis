from numpy import log2
from math import sqrt
import pandas as pd
import numpy as np
import glob
import os


class Data:
    df = pd.DataFrame()
    labels = []

    def __init__(self, data):
        self.df = pd.DataFrame(data)

    def sortByChromosome(self):     # sorts dataset by chromosome number
        self.df.sort_index(inplace=True)

    def geneFiltering(self):        # removes genes with variance less than 1.3
        variance = self.df.var(axis=1)

        for i, row in self.df.iterrows():
            if variance[i] < 1.3:
                self.df.drop(index=i, axis=0, inplace=True)

    def reshapeData(self):         # reshapes data to images
        # calculates the dimensions for the images
        new_size = sqrt(len(self.df))
        if not new_size.is_integer():
            new_size = int(new_size) + 1

        reshaped = []
        zeros = [0] * (new_size ** 2 - self.df.shape[0])

        for i in range(self.df.shape[1]):
            sample = self.df.iloc[:, i].values.tolist()

            # add zeroes to the sample to reach the closest square number
            sample = sample + zeros

            # reshapes sample to image
            sample = np.reshape(a=sample, newshape=(new_size, new_size))
            reshaped.append(sample)

        return reshaped


def loadDataset(dir_path):      # loads all datasets in the directory in a dataframe
    all_files = glob.glob(os.path.join(dir_path, "*.csv"))
    return (pd.read_csv(f, sep=',', index_col=[0], dtype=str) for f in all_files)


def mergeDatasets(datasets):    # merges a list of dataframes
    return pd.concat(datasets, join="inner", ignore_index=True, axis=1)


def scaleReduction(elem):       # reduces gene scale
    return log2(elem + 1)


def noiseReduction(elem):       # reduces gene noise
    if elem < 1:
        return 0
    else:
        return elem


def imageNormalization(elem):   # normalizes image values
    return elem / 255.0
