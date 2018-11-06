"""
.. module:: EmotionPredictor
    :synopsis: Use this script to run classifiers on emotion data
"""

import argparse
import json
import multiprocessing
import os
import pickle
import sys

import dask.dataframe as dd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from OpenFaceScripts.scoring import AUScorer


def make_emotion_data(emotion: str, dict_to_use: dict, ck=True):
    """
    Make emotion data for classifiers

    :param emotion: Emotion to classify
    :param dict_to_use: Location of stored DataFrame
    :param ck: If ck dict exists
    """

    # if dict_to_use is None:
        # dict_to_use = json.load(open('au_emotes.txt'))
    # emotion_data = [
        # item for sublist in (b for b in ((a for a in x.values() if a)
                                         # for x in dict_to_use.values()

                                         # if x) if b) for item in sublist
    # ]

    # if ck:
        # ck_dict = json.load(open('ck_dict.txt'))

        # for patient_list in ck_dict.values():
            # to_add = AUScorer.TrainList
            # au_dict = {
                # str(int(float(x))): y

                # for x, y in patient_list[0].items()
            # }

            # for add in to_add:
                # if add not in au_dict:
                    # au_dict[add] = 0
            # emotion_data.append([au_dict, patient_list[1]])

    # au_data = []
    # target_data = []
    # aus_list = AUScorer.TrainList

    # for frame in emotion_data:
        # aus = frame[0]

        # if frame[1] == emotion:
            # au_data.append([float(aus[str(x)]) for x in aus_list])
            # target_data.append(1)
    # index = 0
    # happy_len = len(target_data)

    # for frame in emotion_data:
        # aus = frame[0]

        # if frame[1] and frame[1] != emotion:
            # au_data.append([float(aus[str(x)]) for x in aus_list])
            # target_data.append(0)
            # index += 1

        # if index == happy_len:
            # break

    # return au_data, target_data


def use_classifier(out_q, emotion: str, classifier, df):
    """
    Train an emotion classifier

    :param out_q: Queue to put classification report in (for multiprocessing)
    :param emotion: emotion to classify
    :param classifier: classifier to train and dump
    """

    # out_q.put(emotion + '\n')
    # out_q.put('Best params \n')
    # out_q.put(str(classifier.best_params_) + '\n')
    # out_q.put("Best f1 score \n")
    # out_q.put(str(classifier.best_score_) + '\n')
    au_data, target_data = make_emotion_data(emotion, df)
    scores = cross_val_score(
        classifier, au_data, target_data, scoring='precision')
    out_q.put(emotion + '\n')
    out_q.put("Cross val precision for classifier {0}:\n{1}\n".format(
        classifier, scores.mean()))
    scores = cross_val_score(
        classifier, au_data, target_data, scoring='recall')
    out_q.put("Cross val recall for classifier {0}:\n{1}\n".format(
        classifier, scores.mean()))
    au_train, au_test, target_train, target_test = train_test_split(
        au_data, target_data, test_size=.1)
    classifier.fit(au_train, target_train)

    expected = target_test
    predicted = classifier.predict(au_test)

    out_q.put("Classification report for classifier %s:\n%s\n" %
              (classifier, metrics.classification_report(expected, predicted)))
    out_q.put("Confusion matrix:\n%s\n" % metrics.confusion_matrix(
        expected, predicted))  # joblib.dump(classifier,
    # '{0}_trained_RandomForest_with_pose'.format(emotion), compress=1)
    pickle.dump(
        classifier,
        open('{0}_trained_RandomForest_with_pose.pkl'.format(emotion), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("OpenDir", help="Path to OpenFaceTests directory")
    parser.add_argument("DataFrame", help="Path to DataFrame to train on")
    args = parser.parse_args()
    OpenDir = args.OpenDir
    df = dd.read_hdf(args.DataFrame, key='data')
    os.chdir(OpenDir)

    # def make_random_forest(emotion) -> GridSearchCV:
    #     param_grid = {
    #         'n_estimators': np.arange(1, 20, 5),
    #         'max_features': ['auto'] + list(np.linspace(.1, 1, 5)),
    #         'max_depth': [None] + list(np.arange(1, 10, 5)),
    #         'min_samples_split': np.linspace(.1, 1, 5),
    #         'min_samples_leaf': np.linspace(.1, .5, 5),
    #         'min_weight_fraction_leaf': np.linspace(0, .5, 5),
    #         'max_leaf_nodes': [None] + list(np.arange(2, 100, 10)),
    # 'min_impurity_split': np.linspace(0, 1, 5),  # Figure out what this does
    #         'bootstrap': [True, False],
    #     }
    #     random_forest = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', n_jobs=multiprocessing.cpu_count(), verbose=5)
    #     au_data, target_data = make_emotion_data(emotion)
    #     au_train, au_test, target_train, target_test = train_test_split(au_data, target_data, test_size=.1)
    #     random_forest.fit(au_train, target_train)
    #     return random_forest

    out_file = open('classifier_performance.txt', 'w')
    out_q = multiprocessing.Manager().Queue()

    index = 1
    # bar = progressbar.ProgressBar(redirect_stdout=True, max_value=1 *
    # len(AUScorer.emotion_list()))

    for emotion in AUScorer.emotion_list():
        classifiers = [
            RandomForestClassifier(),
        ]

        for classifier in classifiers:
            use_classifier(out_q, emotion, classifier, df)
            # bar.update(index)
            index += 1

    while not out_q.empty():
        out_file.write(out_q.get())
