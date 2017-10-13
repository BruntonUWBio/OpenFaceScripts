import copy
import functools
import json
import multiprocessing
import os
import sys
from random import shuffle

import dill
import numpy as np
import progressbar
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from autosklearn import classification

def restricted_k_neighbors(out_q):
    emotion_data = [item for sublist in
                    [b for b in [[a for a in x.values() if a] for x in json.load(open('au_emotes.txt')).values() if x]
                     if b]
                    for item in sublist if item[1] in ['Happy', 'Neutral', 'Sleeping']]
    au_data = []
    target_data = []
    aus_list = sorted([12, 6, 26, 10, 23])
    for frame in emotion_data:
        aus = frame[0]
        if frame[1] == 'Happy':
            au_data.append([float(aus[str(x)]) for x in aus_list])
            target_data.append(frame[1])
    index = 0
    happy_len = len(target_data)
    for frame in emotion_data:
        aus = frame[0]
        if frame[1] != 'Happy':
            au_data.append([float(aus[str(x)]) for x in aus_list])
            target_data.append('Neutral/Sleeping')
            index += 1
        if index == happy_len:
            break
    au_data = np.array(au_data)
    target_data = np.array(target_data)
    n_samples = len(au_data)

    au_data_shuf = []
    target_data_shuf = []
    index_shuf = list(range(len(au_data)))
    shuffle(index_shuf)
    for i in index_shuf:
        au_data_shuf.append(au_data[i])
        target_data_shuf.append(target_data[i])
    au_data = copy.copy(au_data_shuf)
    target_data = copy.copy(target_data_shuf)
    au_data = np.array(au_data)
    target_data = np.array(target_data)
    use_classifier(out_q, au_data, target_data, KNeighborsClassifier())


def use_classifier(out_q, au_data, target_data, classifier):
    nine_tenths = (n_samples // 10) * 9
    classifier.fit(au_data[:nine_tenths], target_data[:nine_tenths])

    expected = target_data[nine_tenths:]
    predicted = classifier.predict(au_data[nine_tenths:])

    out_q.put("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))
    out_q.put("Confusion matrix:\n%s\n" % metrics.confusion_matrix(expected, predicted))

    joblib.dump(classifier, 'happy_trained_RandomForest.pkl')



OpenDir = sys.argv[sys.argv.index('-d') + 1]
os.chdir(OpenDir)
emotion_data = [item for sublist in
                [b for b in [[a for a in x.values() if a] for x in json.load(open('au_emotes.txt')).values() if x] if b]
                for item in sublist if item[1] in ['Happy', 'Neutral', 'Sleeping']]
au_data = []
target_data = []
aus_list = sorted([int(x) for x in emotion_data[0][0].keys()])
for frame in emotion_data:
    aus = frame[0]
    if frame[1] == 'Happy':
        au_data.append([float(aus[str(x)]) for x in aus_list])
        # target_data.append(frame[1])
        target_data.append(1)
index = 0
happy_len = len(target_data)
for frame in emotion_data:
    aus = frame[0]
    if frame[1] != 'Happy':
        au_data.append([float(aus[str(x)]) for x in aus_list])
        # target_data.append('Neutral/Sleeping')
        target_data.append(0)
        index += 1
    if index == happy_len:
        break

n_samples = len(au_data)

au_data_shuf = []
target_data_shuf = []
index_shuf = list(range(len(au_data)))
shuffle(index_shuf)
for i in index_shuf:
    au_data_shuf.append(au_data[i])
    target_data_shuf.append(target_data[i])
au_data = copy.copy(au_data_shuf)
target_data = copy.copy(target_data_shuf)
au_data = np.array(au_data)
target_data = np.array(target_data)

# classifiers = [
#     KNeighborsClassifier(),
#     SVC(kernel='linear'),
#     SVC(),
#     # GaussianProcessClassifier(),
#     # DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     MLPClassifier(),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis(),
# ]

classifiers = [
    RandomForestClassifier(),
]



out_q = multiprocessing.Manager().Queue()
out_file = open('classifier_performance.txt', 'w')
f = functools.partial(use_classifier, out_q, au_data, target_data)
bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(classifiers))
for i, _ in enumerate(Pool().imap(f, classifiers), 1):
    bar.update(i)
# print('auto-sklearn...')
# use_classifier(out_q, au_data, target_data, classification.AutoSklearnClassifier())


# print('restricted_k_neighbors...')
# restricted_k_neighbors(out_q)
while not out_q.empty():
    out_file.write(out_q.get())
