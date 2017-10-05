import functools
import json
import os

import multiprocessing

import progressbar
import sklearn
from sklearn import datasets, svm, metrics, neural_network
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from pathos.multiprocessing import ProcessingPool as Pool
import sys
import numpy as np


def use_classifier(out_q, classifier):
    nine_tenths = (n_samples // 10) * 9
    classifier.fit(au_data[:nine_tenths], target_data[:nine_tenths])

    expected = target_data[nine_tenths:]
    predicted = classifier.predict(au_data[nine_tenths:])

    out_q.put("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    out_q.put("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


OpenDir = sys.argv[sys.argv.index('-d') + 1]
os.chdir(OpenDir)
emotion_data =  [item for sublist in [b for b in [[a for a in x.values() if a] for x in json.load(open('au_emotes.txt')).values() if x] if b] for item in sublist if item[1] in ['Happy', 'Neutral', 'Sleeping']]
au_data = []
target_data = []
aus_list = sorted([int(x) for x in emotion_data[0][0].keys()])
for frame in emotion_data:
    aus = frame[0]
    au_data.append([float(aus[str(x)]) for x in aus_list])
    target_data.append(frame[1]) if frame[1] == 'Happy' else target_data.append('Neutral/Sleeping')
au_data = np.array(au_data)
target_data = np.array(target_data)
n_samples = len(au_data)

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel='linear'),
    SVC(),
    # GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

out_q = multiprocessing.Manager().Queue()
out_file = open('classifier_performance.txt', 'w')
f = functools.partial(use_classifier, out_q)
bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(classifiers))
for i, _ in enumerate(Pool().imap(f, classifiers), 1):
    bar.update(i)
while not out_q.empty():
    out_file.write(out_q.get())