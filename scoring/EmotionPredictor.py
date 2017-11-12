import functools
import json
import multiprocessing
import os
import sys

import numpy as np
import progressbar
from pathos.multiprocessing import ProcessingPool as Pool
from scoring import AUScorer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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

    au_train, au_test, target_train, target_test = train_test_split(au_data, target_data, test_size=.1)
    use_classifier(out_q, au_train, au_test, target_train, target_test, KNeighborsClassifier())


def use_classifier(out_q, au_train, au_test, target_train, target_test, classifier):
    classifier.fit(au_train, target_train)

    expected = target_test
    predicted = classifier.predict(au_test)

    out_q.put("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))
    out_q.put("Confusion matrix:\n%s\n" % metrics.confusion_matrix(expected, predicted))
    joblib.dump(classifier, 'happy_trained_RandomForest_with_pose.pkl')


OpenDir = sys.argv[sys.argv.index('-d') + 1]
os.chdir(OpenDir)
emotion_data = [item for sublist in
                [b for b in [[a for a in x.values() if a] for x in json.load(open('au_emotes.txt')).values() if x] if b]
                for item in sublist if item[1] in ['Happy', 'Neutral', 'Sleeping']]
ck_dict = json.load(open('ck_dict.txt'))
for patient_list in ck_dict.values():
    if patient_list[1] in [None, 'Happy']:
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
    if frame[1] == 'Happy':
        au_data.append([float(aus[str(x)]) if str(x) in aus else 0 for x in aus_list])
        # target_data.append(frame[1])
        target_data.append(1)
index = 0
happy_len = len(target_data)
for frame in emotion_data:
    aus = frame[0]
    if frame[1] != 'Happy':
        au_data.append([float(aus[str(x)]) if str(x) in aus else 0 for x in aus_list])
        # target_data.append('Neutral/Sleeping')
        target_data.append(0)
        index += 1
    if index == happy_len:
        break

n_samples = len(au_data)

au_train, au_test, target_train, target_test = train_test_split(au_data, target_data, test_size=.1)

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
f = functools.partial(use_classifier, out_q, au_train, au_test, target_train, target_test)
bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(classifiers))
for i, _ in enumerate(Pool().imap(f, classifiers), 1):
    bar.update(i)
# print('auto-sklearn...')
# use_classifier(out_q, au_data, target_data, classification.AutoSklearnClassifier())


# print('restricted_k_neighbors...')
# restricted_k_neighbors(out_q)
while not out_q.empty():
    out_file.write(out_q.get())
