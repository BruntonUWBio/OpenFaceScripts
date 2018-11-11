
"""
.. module:: EmotionAppender
    :synopsis: Use an existing classifier to add classification data to a DataFrame
"""
import argparse
import pickle

import dask.dataframe as dd

def add_classification(dataframe_path, dataframe: dd.DataFrame, classifier_path, emotion:str):
    dataframe = dataframe.assign(predicted=lambda x: classifier_path.predict([x for x in x.keys() if 'annotated' not in x and 'predicted' not in x and 'patient' not in x])).compute()
    dataframe.to_hdf(dataframe_path, '/data', format='table')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataframe", help="Path to DataFrame")
    parser.add_argument("classifier", help="Path to classifier")
    parser.add_argument("emotion", help="Emotion")

    ARGS = parser.parse_args()
    dataframe = dd.read_hdf(ARGS.dataframe)
    classifier_path = pickle.load(ARGS.classifier)
    emotion = ARGS.emotion

    add_classification(ARGS.dataframe, dataframe, classifier_path, emotion)
