import json
import os

import sys

import progressbar

sys.path.append('/home/gvelchuru/OpenFaceScripts')
import AUGui
import functools
from scoring import AUScorer
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from runners import VidCropper, SecondRunOpenFace

patient_dir = "/data2/OpenFaceTests"
os.chdir(patient_dir)


def combine_scores():
    all_dicts = {}
    duration_dict = {}
    all_dict_q =  multiprocessing.Manager().Queue()
    duration_dict_q = multiprocessing.Manager().Queue()
    dirs = [y for y in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, y))]
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(dirs))
    f = functools.partial(scores_and_duration_dict, all_dict_q, duration_dict_q)
    for i, _ in enumerate(Pool().imap(f, dirs), 1):
        bar.update(i)
    while not all_dict_q.empty():
        all_dicts.update(all_dict_q.get())
        duration_dict.update(duration_dict_q.get())

    json.dump(all_dicts, open(os.path.join(patient_dir, 'all_dict.txt'), 'w'))
    json.dump(duration_dict, open(os.path.join(patient_dir, 'duration_dict.txt'), 'w'))

def scores_and_duration_dict(all_dict_q, duration_dict_q, dir):
    if 'cropped' in dir:
        all_dicts = {}
        duration_dict = {}
        remove_crop = dir.replace('_cropped', '')
        dir_num = remove_crop[len(remove_crop) - 4:len(remove_crop)]
        patient_name = remove_crop.replace('_' + dir_num, '')
        if patient_name not in all_dicts:
            all_dicts[patient_name] = {}
        if patient_name not in duration_dict:
            duration_dict[patient_name] = {}
        all_dict_file = os.path.join(dir, 'all_dict.txt')
        if os.path.exists(all_dict_file):
            all_dicts[patient_name][int(dir_num)] = json.load(open(all_dict_file))
        else:
            all_dicts[patient_name][int(dir_num)] = AUScorer.AUScorer(dir).emotions
        duration_dict[patient_name][int(dir_num)] = int(VidCropper.duration(SecondRunOpenFace.get_vid_from_dir(dir)) * 30)
        all_dict_q.put(all_dicts)
        duration_dict_q.put(duration_dict)


def make_scatters(scores_dict, duration_dict, patient):
    try:
        patient_scores_dir = patient
        plot_dict = make_scatter_plot_dict(scores_dict[patient])
        if not os.path.exists(patient_scores_dir):
            os.mkdir(patient_scores_dir)
        if 'day_scores.png' not in os.listdir(patient_scores_dir):
            maxKey = max(map(int, scores_dict[patient].keys()))
            maxVal = (maxKey + 1) * 120 * 30
            emotions = plot_dict.keys()
            frame_list = [i for i in range(maxVal)]
            data = np.ndarray(shape=(len(emotions), len(frame_list)))
            for index, emotion in enumerate(emotions):
                new_row = [0 for _ in range(len(frame_list))]
                for vid in plot_dict[emotion]:
                    for frame in plot_dict[emotion][vid]:
                        new_row[int(vid) * 120 * 30 + int(frame)] = plot_dict[emotion][vid][frame]
                z = (list(new_row))
                data[index] = z

            ax = sns.heatmap(data, yticklabels=emotions, xticklabels=False, cmap='Blues')
            ax.set_title(patient)
            fig = ax.get_figure()
            fig.savefig(os.path.join(patient, 'day_scores.png'))
            #print(patient)
            plt.close()
    except ValueError as e:
        pass
    except IndexError:
        pass


def make_scatter_plot_dict(patient_dict: dict) -> dict:
    scatter_plot_dict = {}
    for vid in patient_dict:
        for frame in patient_dict[vid]:
            if patient_dict[vid][frame]:
                for emotion in patient_dict[vid][frame][0]:
                    if emotion not in scatter_plot_dict:
                        scatter_plot_dict[emotion] = {}
                    if vid not in scatter_plot_dict[emotion]:
                        scatter_plot_dict[emotion][vid] = {}
                    scatter_plot_dict[emotion][vid][frame] = patient_dict[vid][frame][1]
    return scatter_plot_dict


if not(os.path.exists('all_dict.txt') and os.path.exists('duration_dict.txt')):
    combine_scores()
all_dicts = json.load(open('all_dict.txt'))
duration_dicts = json.load(open('duration_dict.txt'))
scores_dir = 'Scores'
if not os.path.exists(scores_dir):
    os.mkdir(scores_dir)
os.chdir(scores_dir)
scores_file = 'no_csv_scores.txt'
# TODO: Parallelize


if not os.path.exists(scores_file):
    scores_dict = {}
    for patient in all_dicts:
        scores_dict[patient] = {}
        currPatientDict = all_dicts[patient]
        for vid in currPatientDict:
            scores_dict[patient][vid] = {}
            for frame in currPatientDict[vid]:
                emotionDict = currPatientDict[vid][frame]
                if emotionDict:
                    reverse_emotions = AUScorer.reverse_emotions(emotionDict)
                    max_value = max(reverse_emotions.keys())
                    max_emotions = reverse_emotions[max_value]
                    prevalence_score = AUGui.prevalence_score(emotionDict)
                    scores_dict[patient][vid][frame] = [max_emotions, prevalence_score]
    json.dump(scores_dict, open(scores_file, 'w'))
else:
    scores_dict = json.load(open(scores_file))
#with open('log.txt', 'w') as log:
f = functools.partial(make_scatters, scores_dict, duration_dicts)
bar = progressbar.ProgressBar(redirect_stdout=True, max_value=1)
for i, _ in enumerate(Pool(1).imap(f, scores_dict.keys()), 1):
    bar.update(i / len(scores_dict.keys()))