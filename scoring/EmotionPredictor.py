import functools
import json
import multiprocessing
import os
import sys

import numpy as np
import progressbar
from autosklearn.estimators import AutoSklearnClassifier
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts.scoring import AUScorer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def make_emotion_data(emotion):
    emotion_data = [item for sublist in
                    [b for b in [[a for a in x.values() if a] for x in json.load(open('au_emotes.txt')).values() if x]
                     if b]
                    for item in sublist]

    ck_dict = json.load(open('ck_dict.txt'))
    for patient_list in ck_dict.values():
        to_add = AUScorer.AUList
        au_dict = {str(int(float(x))): y for x, y in patient_list[0].items()}
        for add in to_add:
            if add not in au_dict:
                au_dict[add] = 0
        emotion_data.append([au_dict, patient_list[1]])

    au_data = []
    target_data = []
    aus_list = AUScorer.AUList
    for frame in emotion_data:
        aus = frame[0]
        if frame[1] == emotion:
            au_data.append([float(aus[str(x)]) for x in aus_list])
            target_data.append(1)
    index = 0
    happy_len = len(target_data)
    for frame in emotion_data:
        aus = frame[0]
        if frame[1] and frame[1] != emotion:
            au_data.append([float(aus[str(x)]) for x in aus_list])
            target_data.append(0)
            index += 1
        if index == happy_len:
            break

    return au_data, target_data


def use_classifier(out_q, emotion: str,
                   classifier: GridSearchCV):
    out_q.put(emotion + '\n')
    out_q.put('Best params \n')
    out_q.put(classifier.best_params_ + '\n')

    au_data, target_data = make_emotion_data(emotion)
    # scores = cross_val_score(classifier, au_data, target_data, scoring='precision')
    # out_q.put(emotion + '\n')
    # out_q.put("Cross val precision for classifier {0}:\n{1}\n".format(classifier, scores.mean()))
    # scores = cross_val_score(classifier, au_data, target_data, scoring='recall')
    # out_q.put("Cross val recall for classifier {0}:\n{1}\n".format(classifier, scores.mean()))
    au_train, au_test, target_train, target_test = train_test_split(au_data, target_data, test_size=.1)

    expected = target_test
    predicted = classifier.predict(au_test)

    out_q.put("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))
    out_q.put("Confusion matrix:\n%s\n" % metrics.confusion_matrix(expected, predicted))
    joblib.dump(classifier, '{0}_trained_RandomForest_with_pose.pkl'.format(emotion))


OpenDir = sys.argv[sys.argv.index('-d') + 1]
os.chdir(OpenDir)


def make_random_forest(emotion) -> GridSearchCV:
    param_grid = {
        'n_estimators': np.arange(1, 20),
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto'] + list(np.linspace(0, 1, 10)),
        'max_depth': ['None'] + list(np.arange(1, 10)),
        'min_samples_split': np.linspace(0, 1, 10),
        'min_samples_leaf': np.linspace(.1, .5, 5),
        'min_weight_fraction_leaf': np.linspace(0, 1, 10),
        'max_leaf_nodes': [None] + list(np.arange(1, 100, 10)),
        'min_impurity_split': np.linspace(0, 1, 10),  # Figure out what this does
        'bootstrap': [True, False],
        'oob_score': [True, False]
    }
    # param_grid = ParameterGrid(param_grid)
    # list_grid = list(param_grid)
    random_forest = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', n_jobs=multiprocessing.cpu_count())
    au_data, target_data = make_emotion_data(emotion)
    au_train, au_test, target_train, target_test = train_test_split(au_data, target_data, test_size=.1)
    random_forest.fit(au_train, target_train)
    return random_forest

out_file = open('classifier_performance.txt', 'w')
out_q = multiprocessing.Manager().Queue()

index = 1
bar = progressbar.ProgressBar(redirect_stdout=True, max_value=1 * len(AUScorer.emotion_list()))
for emotion in AUScorer.emotion_list():
    classifiers = [
        make_random_forest(emotion),
    ]
    f = functools.partial(use_classifier, out_q, emotion)
    for i, _ in enumerate(Pool().imap(f, classifiers)):
        bar.update(index)
        index += 1

while not out_q.empty():
    out_file.write(out_q.get())
