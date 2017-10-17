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
from matplotlib.colors import LogNorm, Colormap, LinearSegmentedColormap
from matplotlib.ticker import LogLocator, AutoLocator

import seaborn as sns

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from runners import VidCropper, SecondRunOpenFace

patient_dir = "/data2/OpenFaceTests"
os.chdir(patient_dir)


def combine_scores():
    all_dicts = {}
    duration_dict = {}
    all_dict_q = multiprocessing.Manager().Queue()
    duration_dict_q = multiprocessing.Manager().Queue()
    dirs = [y for y in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, y))]
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(dirs))
    f = functools.partial(scores_and_duration_dict, all_dict_q, duration_dict_q)
    p = Pool()
    for i, _ in enumerate(p.imap(f, dirs, chunksize=50), 1):
        bar.update(i)
    p.close()
    p.join()
    while not all_dict_q.empty():
        patient_dict = all_dict_q.get()
        dur_dict = duration_dict_q.get()
        for i in patient_dict:
            print(i)
            if i not in all_dicts:
                all_dicts[i] = patient_dict[i]
            else:
                all_dicts[i].update(patient_dict[i])
        for i in dur_dict:
            print(i)
            if i not in duration_dict:
                duration_dict[i] = dur_dict[i]
            else:
                duration_dict[i].update(dur_dict[i])
    print('done combining scores, dumping...')
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
        duration_dict[patient_name][int(dir_num)] = int(
            VidCropper.duration(SecondRunOpenFace.get_vid_from_dir(dir)) * 30)
        all_dict_q.put(all_dicts)
        duration_dict_q.put(duration_dict)


def make_scatter_data(patient):
    patient_scores_dir = patient
    plot_dict_file = os.path.join(patient_scores_dir, 'plot_dict.txt')
    if os.path.exists(patient_scores_dir):
        try:
            temp_plot_dict = json.load(open(plot_dict_file)) if os.path.exists(plot_dict_file) else None
        except:
            temp_plot_dict = None
    else:
        temp_plot_dict = None

    #plot_dict = make_scatter_plot_dict(scores_dict[patient])
    plot_dict = scores_dict[patient]

    if not temp_plot_dict or (plot_dict != temp_plot_dict):
        emotions = AUScorer.emotion_list()
        emotions.append('Neutral')
        temp_data = {emotion: [] for emotion in emotions}
        for vid in sorted(plot_dict.keys()):
            for frame in sorted(plot_dict[vid].keys()):
                for emotion in emotions:
                    if emotion in plot_dict[vid][frame][0]:
                        temp_data[emotion].append(plot_dict[vid][frame][1])
                    else:
                        temp_data[emotion].append(0)

        # for index, emotion in enumerate(emotions):
        #     new_row = []
        #     for vid in sorted(plot_dict[emotion].keys()):
        #         for frame in sorted(plot_dict[emotion][vid].keys()):
        #             new_row.append(plot_dict[emotion][vid][frame])
        #     z = (list(new_row))
        #     temp_data[emotion] = z

        data = []
        for index, emotion in enumerate(sorted(x for x in emotions if x != 'Neutral')):
            data.append([x + 0.00000000000000000000000000000001 for x in temp_data[emotion]])


        neutral_data = [0.00000000000000000000000000000001 for _ in range(len(data[0]))]

        emotion_dict = {
            'Angry': .4,
            'Sad': .3,
            'Happy': .3,
            'Disgust': .3,
            'Fear': .5,
            'Surprise': .5
        }

        for index, datum in enumerate(data):
            emotion = emotions[index]
            for index, val in enumerate(datum):
                if val < emotion_dict[emotion]:
                    datum[index] = 0.00000000000000000000000000000001
                    neutral_data[index] = max(val, neutral_data[index])

        data.append(neutral_data)

        # real_data = np.ndarray((len(temp_data),))
        # for index, vals in enumerate(data):
        #     real_data[index] = vals

        if all(data):
            # contains = False
            data = np.asarray(data, dtype=np.float64)
            # for _ in data:
            #     for index, val in enumerate(_):
            #         if val < 1.5:
            #             _[index] = 0
            #         else:
            #             contains = True
            # if contains:
            if not os.path.exists(patient):
                os.mkdir(patient)

            # cict = {
            #     'red': ((0.0, 0.0, 0.0),
            #             (1.0, 0.0, 0.0)),
            #
            #     'green': ((0.0, 0.0, 0.0),
            #               (1.0, 0.0, 0.0)),
            #
            #     'blue': ((0.0, 0.0, 0.0),
            #                  (.25, 0.0, 1.0),
            #                  (1.0, 1.0, 1.0))
            # }
            # test_colormap = matplotlib.colors.ListedColormap('test', cict)

            ax = sns.heatmap(data, cbar_kws={'ticks': LogLocator()}, yticklabels=emotions, xticklabels=False, cmap='BuGn', norm=matplotlib.colors.LogNorm())
            # ax.set_clim(vmin=1.5, vmax=5)
            ax.set_title(patient)
            fig = ax.get_figure()
            fig.savefig(os.path.join(patient, 'day_scores.png'))
            # print(patient)
            plt.close()
            json.dump(plot_dict, open(plot_dict_file, 'w'))


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


if not (os.path.exists('all_dict.txt') and os.path.exists('duration_dict.txt')):
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
    print('making scores file')
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
print('making scatters...')
bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(scores_dict.keys()))
for i, _ in enumerate(Pool().imap(make_scatter_data, sorted(scores_dict.keys()), chunksize=10), 1):
    bar.update(i)
