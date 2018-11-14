"""
.. module:: EmotionAppender
    :synopsis: Use an existing classifier to add classification data to a DataFrame
"""
import argparse
import pickle

import dask
import dask.dataframe as dd
import os
import pandas as pd

from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, parallel_backend
from sklearn.ensemble import RandomForestClassifier

from dask.distributed import Client, LocalCluster


def add_classification(
    dataframe_path,
    dataframe: dd.DataFrame,
    classifier_path: RandomForestClassifier,
    emotion: str,
):
    # cluster = LocalCluster(n_workers=20, ncores=32, processes=False)
    # client = Client(cluster)
    client = Client(processes=False)
    print(client)
    with parallel_backend("dask"):
        # dataframe = dataframe.set_index("patient")
        # predicted = classifier_path.predict(
        # dataframe[
        # [
        # c
        # for c in dataframe.columns
        # if "predicted" not in c
        # and "patient" not in c
        # and "session" not in c
        # and "vid" not in c
        # and "annotated" not in c
        # ]
        # ]
        # )
        # kwargs = {
        # "{0}_predicted".format(emotion): pd.Series(
        # predicted, index=dataframe.index.compute()
        # )
        # }
        dataframe = dataframe.assign(
            predicted=lambda x: classifier_path.predict(
                x[
                    [
                        c
                        for c in x.columns
                        if all(
                            [
                                w not in c
                                for w in [
                                    "predicted",
                                    "patient",
                                    "session",
                                    "vid",
                                    "annotated",
                                ]
                            ]
                        )
                    ]
                ]
            )[0]
        )
    dataframe.to_hdf(dataframe_path, "/data", format="table", scheduler="processes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataframe_folder", help="Path to DataFrame containing folder")
    parser.add_argument("classifier", help="Path to classifier")
    parser.add_argument("emotion", help="Emotion")

    ARGS = parser.parse_args()
    classifier_path = pickle.load(open(ARGS.classifier, "rb"))
    print(classifier_path)
    dataframe = dd.read_hdf(
        os.path.join(ARGS.dataframe_folder, "all_aus", "au_*.hdf"), "/data"
    )
    emotion = ARGS.emotion

    out_folder = os.path.join(
        ARGS.dataframe_folder, "all_aus_with_{0}_predictions".format(emotion)
    )

    os.mkdir(out_folder)

    add_classification(
        os.path.join(out_folder, "au_*.hdf"), dataframe, classifier_path, emotion
    )
