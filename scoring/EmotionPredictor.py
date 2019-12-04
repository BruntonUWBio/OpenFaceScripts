"""
.. module:: EmotionPredictor
    :synopsis: Use this script to run classifiers on emotion data
"""

import argparse
import json
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing.pool import ThreadPool
import os
import gc
import glob
import pickle
import pandas as pd
import functools
from tqdm import tqdm
import sys

import dask
import dask.dataframe as dd
from dask_ml.xgboost import XGBClassifier
from dask_ml.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_validate

from dask.distributed import Client, LocalCluster

from sklearn.externals.joblib import Parallel, parallel_backend
from dask.diagnostics import ProgressBar

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from OpenFaceScripts.scoring import AUScorer

def use_dask_xgb(out_q, emotion, df: dd.DataFrame):
    # data_columns = [x for x in df.columns if 'annotated' not in x and 'predicted' not in x and 'patient' not in x]
    # data = df[data_columns]
    # # data.convert_objects(convert_numeric=True).dropna()
    # data = data.apply(lambda x: pd.to_numeric(x, errors='coerce'),axis=1, meta={'x': 'f8', 'y': 'f8'})
    # data = data.fillna(0)

    # labels = df[df['annotated'] != "N/A"]
    # labels = labels['annotated']
    # # labels = labels.assign(lambda x: 1 if x['annotated'] == emotion else 0)
    # labels = labels.apply(lambda x: 1 if x == emotion else 0, meta={'x': 'f8', 'y': 'f8'})
    # labels = labels.fillna(0)
    # # labels = labels.compute()

    # X_train, X_test, y_train, y_test = train_test_split(data.values, labels.values)

    # classifier = XGBClassifier()
    # scoring = ['precision', 'recall']
    # scores = cross_val_score(
        # classifier, X_train, y_train, scoring=scoring)
    # out_q.put("Cross val precision for classifier {0}:\n{1}\n".format(
        # classifier, scores['precision'].mean()))
    # out_q.put("Cross val recall for classifier {0}:\n{1}\n".format(
        # classifier, scores['recall'].mean()))

    data_columns = [x for x in df.columns if 'predicted' not in x and 'patient' not in x and 'session' not in x and 'vid' not in x]
    df = df[data_columns]
    data = df[df['annotated'] != "N/A"]
    data = data[data['annotated'] != ""]

    emote_data = data[data['annotated'] == emotion]
    non_emote_data = data[data['annotated'] != emotion]

    non_emote_data = non_emote_data.sample(frac=len(emote_data)/len(non_emote_data))

    data = dd.concat([emote_data, non_emote_data], interleave_partitions=True)
    labels = (data['annotated'] == emotion)

    # print(labels.unique().compute())

    del data['annotated']
    print("PERSISTING DATA")
    # data, labels = dask.persist(data, labels)
    # data = client.compute(data)
    # labels = client.compute(labels)
    data = data.compute()
    labels = labels.compute()
    # df2 = dd.get_dummies(data.categorize()).persist()
    # X_train, X_test, y_train, y_test = train_test_split(df2, labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels)
    # X_train, X_test = data.random_split([.9,.1])
    # y_train, y_test = labels.random_split([.9,.1])

    # cluster = LocalCluster(n_workers=16)
    # cluster = LocalCluster()
    # client = Client(cluster)
    # client = Client('scheduler-address:8786', processes=False)
    # classifier = XGBClassifier()
    scoring = ['precision', 'recall']
    print("TRAINING")
    # classifier.fit(X_train, y_train)
    classifier = RandomForestClassifier(n_estimators=100)

    with parallel_backend('dask'):
        # scores = cross_validate(
            # classifier, X_train.values, y_train.values, scoring=scoring)
        scores = cross_validate(
            classifier, X_train, y_train, scoring=scoring, cv=5, return_train_score=True)
    out_q.put("Scores for emotion {0} \n".format(emotion))
    out_q.put("Cross val train precision for classifier {0}:\n{1}\n".format(
        classifier, scores['train_precision'].mean()))
    out_q.put("Cross val train recall for classifier {0}:\n{1}\n".format(
        classifier, scores['train_recall'].mean()))
    out_q.put("Cross val test precision for classifier {0}:\n{1}\n".format(
        classifier, scores['test_precision'].mean()))
    out_q.put("Cross val test recall for classifier {0}:\n{1}\n".format(
        classifier, scores['test_recall'].mean()))

    # expected = y_test.values
    # predicted = classifier.predict(X_test.values)
    print("PREDICTING")
    expected= y_test
    with parallel_backend('dask'):
        classifier.fit(X_train, y_train)

        predicted = classifier.predict(X_test)
    # predicted = classifier.predict(X_test)

        out_q.put("Classification report for classifier %s:\n%s\n" %
                (classifier, metrics.classification_report(expected, predicted)))
        out_q.put("Confusion matrix:\n%s\n" % metrics.confusion_matrix(
            expected, predicted))
    # classifier.save_model('{0}_trained_XGBoost_with_pose')
    pickle.dump(
        classifier,
        open('{0}_trained_RandomForest_with_pose.pkl'.format(emotion), 'wb'))



def make_emotion_data(emotion: str, dict_to_use: dd.DataFrame, ck=True):
    """
    Make emotion data for classifiers

    :param emotion: Emotion to classify
    :param dict_to_use: Location of stored DataFrame
    :param ck: If ck dict exists
    """
    emotion_data = []
    target_data = []

    for row in tqdm(dict_to_use.iterrows(), total = len(dict_to_use.index)):
        index, data = row
        annotated_emote = data["annotated"]

        if annotated_emote == "N/A":
            continue
        all_columns = data.keys()
        columns_to_use = [x for x in all_columns if 'annotated' not in x and 'predicted' not in x and 'patient' not in x]
        # data = data.loc[:, columns_to_use].compute()
        data = data[columns_to_use]
        # data = data.loc[:, df.columns != "annotated" and df.columns != "Happy_predicted"].compute()
        emotion_data.append(data.tolist())

        if annotated_emote == emotion:
            target_data.append(1)
        else:
            target_data.append(0)

    return emotion_data, target_data


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


def use_classifier(out_q, emotion: str, classifier, dfs):
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
    index = 0

    # for patient_dir in tqdm(PATIENT_DIRS):
            # dfs = []
            # try:
                # curr_df = dd.read_hdf(os.path.join(patient_dir, 'hdfs', 'au.hdf'), '/data')
                # curr_df = curr_df[curr_df[' success'] == 1]
                # curr_df = curr_df.compute()

                # if not curr_df.empty and 'annotated' in curr_df.columns and 'frame' in curr_df.columns:
                    # dfs.append(curr_df)
                    # # import pdb; pdb.set_trace()

                    # # if df is None:
                        # # df = curr_df
                    # # else:
                        # # df = df.append(curr_df)

                        # # if i and i % 500 == 0:
                            # # # df = df.compute()
                            # # df.to_hdf('au_*.hdf', '/data', format='table', scheduler='processes')
                    # # out_q.put(curr_df)

                # if index == 500:

    print("TRAINING")

    num_training = 1000
    au_test = []
    target_test = []
    out_q.put(emotion + '\n')

    while len(dfs) >= num_training:
        curr_df = dd.concat(dfs[:num_training], interleave_partitions=True)
        au_data, target_data = make_emotion_data(emotion, curr_df)
        curr_au_train, curr_au_test, curr_target_train, curr_target_test = train_test_split(au_data, target_data, test_size=.1)
        classifier.fit(curr_au_train, curr_target_train)
        scores = cross_val_score(
            classifier, curr_au_train, curr_target_train, scoring='precision')
        out_q.put("Cross val precision for classifier {0}:\n{1}\n".format(
            classifier, scores.mean()))
        scores = cross_val_score(
            classifier, curr_au_train, curr_target_train, scoring='recall')
        out_q.put("Cross val recall for classifier {0}:\n{1}\n".format(
            classifier, scores.mean()))
        au_test.extend(curr_au_test)
        target_test.extend(curr_target_test)
        classifier.n_estimators += 10
        dfs = dfs[num_training:]
        # dfs.clear()
            # except AttributeError as e:
                # print(e)
            # except ValueError as e:
                # print(e)
            # except KeyError as e:
                # print(e)
    curr_df = dd.concat(dfs, interleave_partitions=True)
    au_data, target_data = make_emotion_data(emotion, curr_df)
    curr_au_train, curr_au_test, curr_target_train, curr_target_test = train_test_split(au_data, target_data, test_size=.1)
    classifier.fit(curr_au_train, curr_target_train)
    scores = cross_val_score(
        classifier, curr_au_train, curr_target_train, scoring='precision')
    out_q.put("Cross val precision for classifier {0}:\n{1}\n".format(
        classifier, scores.mean()))
    scores = cross_val_score(
        classifier, curr_au_train, curr_target_train, scoring='recall')
    out_q.put("Cross val recall for classifier {0}:\n{1}\n".format(
        classifier, scores.mean()))
    au_test.extend(curr_target_train)
    target_test.extend(curr_target_test)


    # if dfs:
        # curr_df = dd.concat(dfs)
        # au_data, target_data = make_emotion_data(emotion, curr_df)
        # classifier.fit(au_data, target_data)
        # dfs.clear()


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

def load_patient(out_q, patient_dir):
        try:
            curr_df = dd.read_hdf(os.path.join(patient_dir, 'hdfs', 'au.hdf'), '/data')
            curr_df = curr_df[curr_df[' success'] == 1]
            # curr_df = curr_df.compute()

            if len(curr_df) and 'annotated' in curr_df.columns and 'frame' in curr_df.columns:
                # import pdb; pdb.set_trace()

                # if df is None:
                    # df = curr_df
                # else:
                    # df = df.append(curr_df)

                    # if i and i % 500 == 0:
                        # # df = df.compute()
                        # df.to_hdf('au_*.hdf', '/data', format='table', scheduler='processes')
                out_q.put(curr_df)

                # return curr_df

        except AttributeError as e:
            print(e)
        except ValueError as e:
            print(e)
        except KeyError as e:
            print(e)

def dump_queue(queue):
    result = []

    while not queue.empty():
        i = queue.get()
        result.append(i)
    # time.sleep(.1)

    return result


if __name__ == '__main__':
    # dask.config.set(pool=ThreadPool(4))
    # cluster = LocalCluster(silence_logs=0)
    parser = argparse.ArgumentParser()
    parser.add_argument("OpenDir", help="Path to OpenFaceTests directory")
    parser.add_argument("--refresh", help="Refresh DataFrame", action="store_true")
    args = parser.parse_args()
    OpenDir = args.OpenDir
    os.chdir(OpenDir)
    au_path = os.path.join('all_aus', 'au_*.hdf')

    if args.refresh:

        if not os.path.exists('all_aus'):
            os.mkdir('all_aus')
        PATIENT_DIRS = [
            x for x in glob.glob('*cropped') if 'hdfs' in os.listdir(x)
        ]
        dfs = []
        df = None
        patient_queue = multiprocessing.Manager().Queue()
        partial_patient_func = functools.partial(load_patient, patient_queue)
        with Pool() as p:
            max_ = len(PATIENT_DIRS)
            with tqdm(total=max_) as pbar:
                for i, _ in enumerate(p.imap(partial_patient_func, PATIENT_DIRS[:max_], chunksize=100)):
                    pbar.update()

        df = dd.concat(dump_queue(patient_queue), interleave_partitions=True)
        del patient_queue
        gc.collect()

        # print(dd.stack(dfs))
        # df = df.compute()
        df.to_hdf(au_path, '/data', format='table', scheduler='processes') #need scheduler to avoid segfault
        print('DUMPED')
        client = Client(processes=False)
        print(client)

    else:
        # df = dd.read_hdf(os.path.join('all_aus', 'au_*.hdf'), '/data')
        # client = Client('tcp://127.0.0.1:46619', processes=False)
        # cluster = LocalCluster(n_workers=30, threads_per_worker=1)
        # cluster = LocalCluster(processes=False)
        # client = Client(cluster)
        # print(client)
        client = Client(processes=False)
        print(client)
        df = dd.read_hdf(au_path, '/data')

    # print("NUM_FRAMES IS" + str(len(df.index)))

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
        # classifiers = [
            # RandomForestClassifier(warm_start=True),
        # ]

        # for classifier in classifiers:
            # use_classifier(out_q, emotion, classifier, dump_queue(patient_queue))
            # # bar.update(index)
            # index += 1
        use_dask_xgb(out_q, emotion, df)

    while not out_q.empty():
        out_file.write(out_q.get())
