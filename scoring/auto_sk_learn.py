import json
import multiprocessing
import os
import sys

import numpy as np
import progressbar
from tpot import TPOTClassifier

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts.scoring import AUScorer
from sklearn import metrics
from sklearn.model_selection import train_test_split


def make_emotion_data(emotion):
    emotion_data = [item for sublist in
                    [b for b in [[a for a in x.values() if a] for x in json.load(open('au_emotes.txt')).values() if x]
                     if b]
                    for item in sublist]

    ck_dict = json.load(open('ck_dict.txt'))
    for patient_list in ck_dict.values():
        to_add = AUScorer.TrainList
        au_dict = {str(int(float(x))): y for x, y in patient_list[0].items()}
        for add in to_add:
            if add not in au_dict:
                au_dict[add] = 0
        emotion_data.append([au_dict, patient_list[1]])

    au_data = []
    target_data = []
    aus_list = AUScorer.TrainList
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

    au_train, au_test, target_train, target_test = train_test_split(au_data, target_data, test_size=.1)
    return au_train, au_test, target_train, target_test


def use_classifier(out_q, au_train: list, au_test: list, target_train: list, target_test: list, emotion: str,
                   classifier):
    training_data = np.array(au_train)
    training_targets = np.array(au_test)
    classifier.fit(np.array(au_train), np.array(target_train))

    expected = target_test
    predicted = classifier.predict(np.array(au_test))
    out_q.put(emotion + '\n')
    out_q.put("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))
    out_q.put("Confusion matrix:\n%s\n" % metrics.confusion_matrix(expected, predicted))
    classifier.export('{0}_trained_auto_sk_learn_with_pose.py'.format(emotion))


OpenDir = sys.argv[sys.argv.index('-d') + 1]
os.chdir(OpenDir)

classifiers = [
    TPOTClassifier(verbosity=3, n_jobs=-1, scoring='f1', memory='auto'),
]
out_file = open('auto_classifier_performance.txt', 'w')
out_q = multiprocessing.Manager().Queue()

index = 1
bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(classifiers) * len(AUScorer.emotion_list()))
for emotion in ['Happy', 'Angry', 'Fear', 'Sad', 'Surprise', 'Disgust']:
    au_train, au_test, target_train, target_test = make_emotion_data(emotion)
    for classifier in classifiers:
        use_classifier(out_q, au_train, au_test, target_train, target_test, emotion, classifier)
        bar.update(index)
        index += 1

while not out_q.empty():
    out_file.write(out_q.get())
