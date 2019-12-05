"""
.. module:: EmotionAppender
    :synopsis: Use an existing classifier to add classification data to a DataFrame
"""
import argparse
import pickle

import dask
import dask.dataframe as dd
import dask.array as da
import os
import pandas as pd

from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, parallel_backend
from sklearn.ensemble import RandomForestClassifier

from dask.distributed import Client, LocalCluster

import numpy as np
import glob
from tqdm import tqdm
import warnings


def predict(x, classifier_function):
    x = x[
        [
            c
            for c in x.columns
            if all(
                [
                    w not in c
                    for w in ["predicted", "patient", "session", "vid", "annotated"]
                ]
            )
        ]
    ]

    # compute_arr = x.compute()
    compute_arr = x
    predicted = classifier_function(compute_arr)

    return predicted


def add_classification(
    dataframe_path, classifier_path: RandomForestClassifier, emotion: str
):
    client = Client(processes=False)
    print(client)

    with parallel_backend("dask"):
        PATIENT_DIRS = [
            x
            for x in glob.glob(os.path.join(dataframe_path, "*cropped"))
            if "hdfs" in os.listdir(x)
        ]

        for patient_dir in tqdm(PATIENT_DIRS):
            try:
                curr_df = dd.read_hdf(
                    os.path.join(patient_dir, "hdfs", "au.hdf"), "/data"
                )
                # curr_df = curr_df[curr_df[" success"] == 1]
                curr_df = curr_df.compute()

                if (
                    len(curr_df)
                    and "annotated" in curr_df.columns
                    and "frame" in curr_df.columns
                ):
                    kwargs = {
                        "{0}_predicted".format(emotion): lambda x: predict(
                            x, classifier_path.predict
                        ),
                        "{0}_predicted_proba".format(emotion): lambda x: [
                            n[1] for n in predict(x, classifier_path.predict_proba)
                        ],
                    }
                    imp_columns=['patient','success','frame','timestamp','annotated','confidence','session','vid','datetime']
                    
                    #curr_df = curr_df.assign(**kwargs)
                    emotion_df = curr_df[imp_columns]
                    emotion_df = emotion_df.assign(**kwargs)
                    # create name for new dataframe (using patient_session_frame)
                    
                    
                    # store in the out_fullpath
                    emotion_df.to_hdf(
                        os.path.join(patient),
                        "/data",
                        format="table",
                        scheduler="processes",
                    )
                else:
                    print(patient_dir + "HAS A PROBLEM")

            except AttributeError as e:
                print(e)
            except ValueError as e:
                print(e)
            except KeyError as e:
                print(e)

        # dataframe = dataframe.compute().assign(
        # predicted=lambda x: predict(x, classifier_path)
        # )
    # dataframe.to_hdf(dataframe_path, "/data", format="table", scheduler="processes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataframe_folder", help="Path to DataFrame containing folder")
    parser.add_argument("classifier", help="Path to classifier")
    parser.add_argument("emotion", help="Emotion")
    parser.add_argument("out_subfolder", help="Sub folder to store the emotion predictions (it will be stored under dataframe_folder directory)")
    

    ARGS = parser.parse_args()
    classifier_path = pickle.load(open(ARGS.classifier, "rb"))
    print(classifier_path)
    # dataframe = dd.read_hdf(
    # os.path.join(ARGS.dataframe_folder, "all_aus", "au_*.hdf"), "/data"
    # )
    emotion = ARGS.emotion

    out_fullpath = os.path.join(
        ARGS.daframe_folder, out_subfolder "all_aus_with_{0}_predictions".format(emotion)
    )
    if ARGS.out_subfolder is not None:
        out_fullpath = os.path.join(
            ARGS.daframe_folder, out_subfolder )
    else: 
        out_fullpath = os.path.join(
            ARGS.daframe_folder,"all_aus_with_{0}_predictions".format(emotion))
    )
    

    if os.path.exists(out_fullpath):
        warnings.warn("You are going to overwrite the emotion dataframes in {0}".format(out_fullpath).)
    else:
        os.mkdir(out_folder)
        
    

    add_classification(ARGS.dataframe_folder, classifier_path, emotion)
