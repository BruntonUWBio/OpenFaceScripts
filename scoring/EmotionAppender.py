
"""
.. module:: EmotionAppender
    :synopsis: Use an existing classifier to add classification data to a DataFrame
"""
import argparse
import pickle

import dask.dataframe as dd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataframe", help="Path to DataFrame")
    parser.add_argument("classifier", help="Path to classifier")

    ARGS = parser.parse_args()
    dataframe = dd.read_hdf(ARGS.dataframe)
    classifier_path = pickle.load(ARGS.classifier)
