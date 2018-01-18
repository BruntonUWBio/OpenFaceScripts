import glob
import json
import sys

import os

from collections import defaultdict

import numpy as np
from numpy import mean
from progressbar import ProgressBar
from sklearn.metrics import precision_score, recall_score

sys.path.append('/home/gvelchuru/OpenFaceScripts')
from datavis.ml_pr_vis import make_scores_file, clean_csv, make_emotion_data
from scoring.AUScorer import emotion_list
import models.tpot


def vis(out_file, short_patient_list, emotion: str):
    out_file.write(emotion + '\n')
    precisions = []
    recalls = []
    for short_patient in short_patient_list:
        OpenDir = sys.argv[sys.argv.index('-d') + 1]
        os.chdir(OpenDir)
        au_train, au_test, target_train, target_test = make_emotion_data(emotion, short_patient)
        if not au_test:
            continue
        pipeline = getattr(models.tpot, emotion.lower() + '_pipeline')
        classifier = pipeline()
        au_train = np.array(au_train)
        target_train = np.array(target_train)
        au_test = np.array(au_test)
        target_test = np.array(target_test)
        classifier.fit(au_train, target_train)
        predicted = classifier.predict(au_test)
        precisions.append(precision_score(target_test, predicted))
        recalls.append(recall_score(target_test, predicted))
    out_file.write(str(precisions) + '\n')
    out_file.write('Average precision score' + str(mean(precisions)))
    out_file.write(str(recalls) + '\n')
    out_file.write('Average recall score' + str(mean(recalls)))


if __name__ == '__main__':
    OpenDir = sys.argv[sys.argv.index('-d') + 1]
    os.chdir(OpenDir)
    au_emote_dict = json.load(open('au_emotes.txt'))
    patient_dirs = glob.glob('*cropped')  # Directories have been previously cropped by CropAndOpenFace
    scores = defaultdict()
    scores_file = 'predic_substring_dict.txt'
    if not os.path.exists(scores_file):
        make_scores_file(scores_file, patient_dirs)
    scores = json.load(open(scores_file))
    csv_file = json.load(open('scores.txt'))
    csv_file = clean_csv(csv_file)
    short_patient_list = set()
    for direc in csv_file:
        short_direc = direc[:direc.index('_')]
        short_patient_list.add(short_direc)
    out_file = open('tpot_scores.txt', 'w')
    out_file.write(str(short_patient_list) + '\n')
    emotion_list = emotion_list()
    bar = ProgressBar(max_value=len(emotion_list))
    for i, emotion in enumerate(emotion_list, 1):
        vis(out_file, short_patient_list, emotion)
        bar.update(i)
