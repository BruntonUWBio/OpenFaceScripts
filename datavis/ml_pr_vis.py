import copy
import functools
import glob
import json
import sys

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_recall_curve

sys.path.append('/home/gvelchuru/OpenFaceScripts')
from scoring import AUScorer

import multiprocessing
import numpy as np
import os
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
from random import shuffle
from progressbar import ProgressBar
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

all_emotions = AUScorer.emotion_list()
all_emotions.extend(['Neutral', 'Sleeping'])


def use_classifier(classifier):
    nine_tenths = (n_samples // 10) * 9
    classifier.fit(au_data[:nine_tenths], target_data[:nine_tenths])
    expected = target_data[nine_tenths:]
    predicted = classifier.predict(au_data[nine_tenths:])
    decision_function = classifier.predict_proba(au_data[nine_tenths:])[:, 1]
    return expected, decision_function


def thresh_calc(out_q, short_patient, thresh):
    curr_dict = {
        thresh: {emotion: {'true_neg': 0, 'false_neg': 0, 'true_pos': 0, 'false_pos': 0} for emotion in all_emotions}}
    for patient in [x for x in scores if x in csv_file and short_patient in x]:
        for vid in scores[patient]:
            curr_vid_dict = scores[patient][vid]
            csv_vid_dict = csv_file[patient][vid]
            for frame in curr_vid_dict:
                if frame in csv_vid_dict and csv_vid_dict[frame]:
                    actual = csv_vid_dict[frame][2]
                else:
                    actual = None
                if actual:
                    if actual in curr_vid_dict[frame]:
                        score = curr_vid_dict[frame][actual]
                    else:
                        score = 0
                    # curr_dict[thresh][actual]['total_pos'] += 1 Try labeling components
                    for other_emotion in (x for x in curr_dict[thresh] if x != actual):
                        if other_emotion in curr_vid_dict[frame]:
                            other_score = curr_vid_dict[frame][other_emotion]
                        else:
                            other_score = 0
                        if other_score >= thresh:
                            curr_dict[thresh][other_emotion]['false_pos'] += 1
                        else:
                            curr_dict[thresh][other_emotion]['true_neg'] += 1
                    if score >= thresh:
                        curr_dict[thresh][actual]['true_pos'] += 1
                    else:
                        curr_dict[thresh][actual]['false_neg'] += 1


                        # predict_emotions = predict_dict['Max']
                        # if actual and actual not in ['Neutral', 'Sleeping']:
                        #     curr_dict[thresh][actual]['total_pos'] += 1
                        #     if actual in predict_emotions and score >= thresh:
                        #         curr_dict[thresh][actual]['true_pos'] += 1
                        # else:
                        #     for emotion in curr_dict[thresh]:
                        #         curr_dict[thresh][emotion]['total_neg'] += 1
                        #         if score >= thresh:
                        #             curr_dict[thresh][emotion]['false_pos'] += 1
    out_q.put(curr_dict)


def clean_csv(csv_file):
    out_dict = {}
    for direc in csv_file:
        remove_crop = direc.replace('_cropped', '')
        dir_num = remove_crop[len(remove_crop) - 4:len(remove_crop)]
        patient_name = remove_crop.replace('_' + dir_num, '')
        if patient_name not in out_dict:
            out_dict[patient_name] = {}
        out_dict[patient_name][str(int(dir_num))] = csv_file[direc]
    # short_patient_dict = {}
    # for direc in out_dict:
    #     short_direc = direc[:direc.index('_')]
    #     if short_direc not in short_patient_dict:
    #         short_patient_dict[short_direc] = 0
    # for direc in out_dict:
    #     short_direc = direc[:direc.index('_')]
    #     short_patient_dict[short_direc] += out_dict[direc]
    # return short_patient_dict
    return out_dict


def validate_thresh_dict(thresh_dict):
    thresh_list = sorted(thresh_dict.keys())
    for index, thresh in enumerate(thresh_list):
        if index:
            prev_thresh = thresh_list[index - 1]
            assert thresh > prev_thresh
            for emotion in thresh_dict[thresh]:

                total_pos = thresh_dict[thresh][emotion]['true_pos'] + thresh_dict[thresh][emotion]['false_neg']
                prev_total_pos = thresh_dict[prev_thresh][emotion]['true_pos'] + thresh_dict[prev_thresh][emotion][
                    'false_neg']
                assert total_pos == prev_total_pos

                total_neg = thresh_dict[thresh][emotion]['false_pos'] + thresh_dict[thresh][emotion]['true_neg']
                prev_total_neg = thresh_dict[prev_thresh][emotion]['false_pos'] + thresh_dict[prev_thresh][emotion][
                    'true_neg']
                assert total_neg == prev_total_neg

                # false positive decreases, true negative increases
                assert thresh_dict[thresh][emotion]['false_pos'] <= thresh_dict[prev_thresh][emotion]['false_pos']

                # true positive decreases, false negative increases
                assert thresh_dict[thresh][emotion]['true_pos'] <= thresh_dict[prev_thresh][emotion]['true_pos']

                # assert that recall is monotonically decreasing
                if total_pos:
                    assert thresh_dict[thresh][emotion]['true_pos'] / total_pos <= thresh_dict[prev_thresh][emotion][
                                                                                       'true_pos'] / prev_total_pos


if __name__ == '__main__':
    OpenDir = sys.argv[sys.argv.index('-d') + 1]
    os.chdir(OpenDir)
    patient_dirs = glob.glob('*cropped')  # Directories have been previously cropped by CropAndOpenFace
    scores = defaultdict()
    scores_file = 'old_all_dict.txt'
    if os.path.exists(scores_file):
        scores = json.load(open(scores_file))
    csv_file = json.load(open('scores.txt'))
    csv_file = clean_csv(csv_file)
    short_patient_list = set()
    for direc in csv_file:
        short_direc = direc[:direc.index('_')]
        short_patient_list.add(short_direc)

    for short_patient in short_patient_list:
        thresh_file = short_patient + '_threshes.txt'
        thresh_dict = json.load(open(thresh_file)) if os.path.exists(thresh_file) else {}
        if not thresh_dict:
            out_q = multiprocessing.Manager().Queue()
            threshes = np.linspace(0, 1.5, 100)
            bar = ProgressBar(max_value=len(threshes))
            f = functools.partial(thresh_calc, out_q, short_patient)
            for i, _ in enumerate(Pool().imap(f, threshes, chunksize=10)):
                while not out_q.empty():
                    thresh_dict.update(out_q.get())
                bar.update(i)
            json.dump(thresh_dict, open(thresh_file, 'w'))
        validate_thresh_dict(thresh_dict)

        for emotion in ['Happy', 'Angry', 'Sad', 'Disgust']:
            # precision-recall
            out_vals = {}
            for thresh in sorted(thresh_dict.keys()):
                if emotion in thresh_dict[thresh]:
                    curr_emote_dict = thresh_dict[thresh][emotion]
                    false_pos = curr_emote_dict['false_pos']
                    true_pos = curr_emote_dict['true_pos']
                    false_neg = curr_emote_dict['false_neg']
                    total_pos = true_pos + false_neg
                    if total_pos and (false_pos + true_pos):
                        precision = true_pos / (false_pos + true_pos)
                        recall = true_pos / total_pos
                        out_vals[thresh] = [precision, recall]
            x_vals = [out_vals[thresh][0] for thresh in sorted(out_vals.keys())]
            y_vals = [out_vals[thresh][1] for thresh in sorted(out_vals.keys())]
            z_vals = [float(x) for x in sorted(out_vals.keys())]

            if x_vals and y_vals and len(x_vals) == len(y_vals):
                fig = plt.figure()
                ax = fig.gca()
                ax.plot(x_vals, y_vals, label='Substring')
                # ml_dict = {
                #     'GaussianNB': [.80, .69],
                #     'QuadraticDiscriminantAnalysis': [.80, .76],
                #     'AdaBoostClassifier': [.80, .73],
                #     'MLPClassifier': [.88, .92],
                #     'SVCLinear': [.87, .57],
                #     'KNeighbors': [.89, .92],
                #     'SVC': [.85, .83],
                #     'KNeighborsSubstring': [.81, .64]
                # }
                # for label in ml_dict:
                #     ax.plot(ml_dict[label][0], ml_dict[label][1], 'o', label=label)

                OpenDir = sys.argv[sys.argv.index('-d') + 1]
                os.chdir(OpenDir)
                emotion_data = [item for sublist in
                                [b for b in
                                 [[a for a in x.values() if a] for x in json.load(open('au_emotes.txt')).values() if x]
                                 if
                                 b]
                                for item in sublist if item[1] in [emotion, 'Neutral', 'Sleeping']]
                au_data = []
                target_data = []
                aus_list = sorted([int(x) for x in emotion_data[0][0].keys()])
                for frame in emotion_data:
                    aus = frame[0]
                    if frame[1] == emotion:
                        au_data.append([float(aus[str(x)]) for x in aus_list])
                        # target_data.append(frame[1])
                        target_data.append(1)
                index = 0
                happy_len = len(target_data)
                for frame in emotion_data:
                    aus = frame[0]
                    if frame[1] != emotion:
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

                classifier_dict = {
                    KNeighborsClassifier(): 'KNeighbors',
                    SVC(kernel='linear', probability=True): 'SVCLinear',
                    SVC(probability=True): 'SVC',
                    # GaussianProcessClassifier(),
                    # DecisionTreeClassifier(),
                    RandomForestClassifier(): 'RandomForest',
                    ExtraTreesClassifier(): 'ExtraTrees',
                    MLPClassifier(): 'MLP',
                    AdaBoostClassifier(): 'AdaBoost',
                    GaussianNB(): 'GaussianNB',
                    QuadraticDiscriminantAnalysis(): 'QuadraticDiscriminantAnalysis',
                    BernoulliNB(): 'BernoulliNB'
                }

                for classifier in classifier_dict.keys():
                    expected, decision_function = use_classifier(classifier)
                    precision, recall, thresholds = precision_recall_curve(expected, decision_function)
                    ax.plot(precision, recall, label=classifier_dict[classifier])

                ax.set_title(
                    'Performance of Different Methods for' + "\' " + emotion + " \'" + 'Recognition from Continuous AUs')
                ax.set_xlabel('Precision')
                ax.set_ylabel('Recall')
                ax.legend()
                plt.savefig(short_patient + '_{0}_pr_with_ML'.format(emotion))
                plt.close()

                # plt.show()
