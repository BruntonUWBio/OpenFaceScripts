import functools
import glob
import json
import sys
from scoring import AUScorer

import multiprocessing
import numpy as np
import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from collections import defaultdict

from progressbar import ProgressBar
from pathos.multiprocessing import ProcessingPool as Pool

all_emotions = AUScorer.emotion_list()
all_emotions.extend(['Neutral', 'Sleeping'])

def thresh_calc(out_q, thresh):
    curr_dict = {thresh: {emotion: {'true_neg': 0, 'false_neg': 0, 'true_pos': 0, 'false_pos': 0} for emotion in all_emotions}}
    for patient in [x for x in scores if x in csv_file]:
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
                        #curr_dict[thresh][actual]['total_pos'] += 1 Try labeling components
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
    for dir in csv_file:
        remove_crop = dir.replace('_cropped', '')
        dir_num = remove_crop[len(remove_crop) - 4:len(remove_crop)]
        patient_name = remove_crop.replace('_' + dir_num, '')
        if patient_name not in out_dict:
            out_dict[patient_name] = {}
        out_dict[patient_name][str(int(dir_num))] = csv_file[dir]
    return out_dict


def validate_thresh_dict(thresh_dict):
    thresh_list = sorted(thresh_dict.keys())
    for index, thresh in enumerate(thresh_list):
        if index:
            prev_thresh = thresh_list[index - 1]
            assert thresh > prev_thresh
            for emotion in thresh_dict[thresh]:

                total_pos = thresh_dict[thresh][emotion]['true_pos'] + thresh_dict[thresh][emotion]['false_neg']
                prev_total_pos = thresh_dict[prev_thresh][emotion]['true_pos'] + thresh_dict[prev_thresh][emotion]['false_neg']
                assert total_pos == prev_total_pos

                total_neg = thresh_dict[thresh][emotion]['false_pos'] + thresh_dict[thresh][emotion]['true_neg']
                prev_total_neg = thresh_dict[prev_thresh][emotion]['false_pos'] + thresh_dict[prev_thresh][emotion]['true_neg']
                assert  total_neg == prev_total_neg

                #false positive decreases, true negative increases
                assert thresh_dict[thresh][emotion]['false_pos'] <= thresh_dict[prev_thresh][emotion]['false_pos']

                #true positive decreases, false negative increases
                assert thresh_dict[thresh][emotion]['true_pos'] <= thresh_dict[prev_thresh][emotion]['true_pos']

                #assert that recall is monotonically decreasing
                if total_pos:
                    assert thresh_dict[thresh][emotion]['true_pos']/total_pos <= thresh_dict[prev_thresh][emotion]['true_pos']/prev_total_pos



if __name__ == '__main__':
    OpenDir = sys.argv[sys.argv.index('-d') + 1]
    os.chdir(OpenDir)
    patient_dirs = glob.glob('*cropped')  # Directories have been previously cropped by CropAndOpenFace
    scores = defaultdict()
    scores_file = 'all_dict.txt'
    if os.path.exists(scores_file):
        scores = json.load(open(scores_file))
    csv_file = json.load(open('scores.txt'))
    csv_file = clean_csv(csv_file)
    thresh_file = 'threshes.txt'
    thresh_dict = json.load(open(thresh_file)) if os.path.exists(thresh_file) else {}
    if not thresh_dict:
        out_q = multiprocessing.Manager().Queue()
        threshes = np.linspace(0, 1.5, 100)
        bar = ProgressBar(max_value=len(threshes))
        f = functools.partial(thresh_calc, out_q)
        for i,_ in enumerate(Pool().imap(f, threshes, chunksize=10)):
            while not out_q.empty():
                thresh_dict.update(out_q.get())
            bar.update(i)
        json.dump(thresh_dict, open(thresh_file, 'w'))
    validate_thresh_dict(thresh_dict)

    for emotion in all_emotions:

        x_vals = []
        y_vals = []
        for thresh in sorted(thresh_dict.keys()):
            if emotion in thresh_dict[thresh]:
                curr_emote_dict = thresh_dict[thresh][emotion]
                total_neg = curr_emote_dict['false_pos'] + curr_emote_dict['true_neg']
                total_pos = curr_emote_dict['true_pos'] + curr_emote_dict['false_neg']
                if total_neg:
                    x_vals.append(thresh_dict[thresh][emotion]['false_pos'] / total_neg)
                if total_pos:
                    y_vals.append(thresh_dict[thresh][emotion]['true_pos'] / total_pos)

        if x_vals and y_vals and len(x_vals) == len(y_vals):
            fig = plt.figure()
            z_vals = sorted(thresh_dict.keys())
            x_vals = list(map(float, x_vals))
            y_vals = list(map(float, y_vals))
            z_vals = list(map(float, z_vals))

            ax = fig.gca()
            ax.plot(x_vals, y_vals)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            plt.savefig('{0}_roc'.format(emotion))

            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # ax.plot(x_vals, z_vals, y_vals)

        #precision-recall
        out_vals = {}
        for thresh in sorted(thresh_dict.keys()):
            if emotion in thresh_dict[thresh]:
                curr_emote_dict = thresh_dict[thresh][emotion]
                false_pos = curr_emote_dict['false_pos']
                true_pos = curr_emote_dict['true_pos']
                false_neg = curr_emote_dict['false_neg']
                total_pos = true_pos + false_neg
                if total_pos and (false_pos + true_pos):
                    precision = true_pos/(false_pos + true_pos)
                    recall = true_pos/total_pos
                    out_vals[thresh] = [precision, recall]
        x_vals = [out_vals[thresh][0] for thresh in sorted(out_vals.keys())]
        y_vals = [out_vals[thresh][1] for thresh in sorted(out_vals.keys())]
        z_vals = [float(x) for x in sorted(out_vals.keys())]


        if x_vals and y_vals and len(x_vals) == len(y_vals):
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(x_vals, y_vals)
            ax.set_xlabel('Precision')
            ax.set_ylabel('Recall')
            plt.savefig('{0}_pr'.format(emotion))
            # plt.close()

        # precision-recall_3D
        if x_vals and y_vals and z_vals and len(x_vals) == len(y_vals) == len(z_vals):
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot(x_vals, z_vals, y_vals)
            ax.set_xlabel('Precision')
            ax.set_zlabel('Recall')
            ax.set_ylabel('Threshold')
            plt.savefig('{0}_pr_3d'.format(emotion))
            # plt.close()

            # if x_vals and y_vals and z_vals:
            #     fig = plt.figure()
            #     ax = fig.gca(projection='3d')
            #     ax.plot(x_vals, z_vals, y_vals)
            #     ax.set_xlabel('Precision')
            #     ax.set_ylabel('Threshold')
            #     ax.set_zlabel('Recall')
            #     ax.set_title(emotion + '_pr')

    plt.show()
