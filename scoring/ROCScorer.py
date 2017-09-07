import functools
import glob
import json
import sys
from scoring import AUScorer

import multiprocessing
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from collections import defaultdict

from progressbar import ProgressBar
from pathos.multiprocessing import ProcessingPool as Pool

all_emotions = AUScorer.emotion_list()
all_emotions.extend(['Neutral', 'Sleeping'])

def thresh_calc(out_q, thresh):
    curr_dict = {thresh: {emotion: {'total_pos': 0, 'total_neg': 0, 'true_pos': 0, 'false_pos': 0} for emotion in all_emotions}}
    for patient in [x for x in scores if x in csv_file]:
        q = csv_file
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
                        curr_dict[thresh][actual]['total_pos'] += 1
                        for emotion in curr_dict[thresh]:
                            if emotion != actual:
                                curr_dict[thresh][emotion]['total_neg'] += 1
                                if emotion in curr_vid_dict[frame]:
                                    score = curr_vid_dict[frame][emotion]
                                else:
                                    score = 0
                                if score >= thresh:
                                    curr_dict[thresh][emotion]['false_pos'] += 1
                        if score >= thresh:
                            curr_dict[thresh][actual]['true_pos'] += 1




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
                assert thresh_dict[thresh][emotion]['total_pos'] == thresh_dict[prev_thresh][emotion]['total_pos']
                assert thresh_dict[thresh][emotion]['total_neg'] == thresh_dict[prev_thresh][emotion]['total_neg']

                #false positive decreases, true negative increases
                assert thresh_dict[thresh][emotion]['false_pos'] <= thresh_dict[prev_thresh][emotion]['false_pos']

                #true positive decreases, false negative increases
                assert thresh_dict[thresh][emotion]['true_pos'] <= thresh_dict[prev_thresh][emotion]['true_pos']

                #assert that recall is monotonically decreasing
                if thresh_dict[thresh][emotion]['total_pos']:
                    assert thresh_dict[thresh][emotion]['true_pos']/thresh_dict[thresh][emotion]['total_pos'] <= thresh_dict[prev_thresh][emotion]['true_pos']/thresh_dict[prev_thresh][emotion]['total_pos']

                # #assert that precision is monotonically increasing
                # assert thresh_dict[thresh][emotion]['true_pos']/(thresh_dict[thresh][emotion]['true_pos'] + thresh_dict[thresh][emotion]['false_pos']) >= thresh_dict[prev_thresh][emotion]['true_pos']/(thresh_dict[prev_thresh][emotion]['true_pos'] + thresh_dict[prev_thresh][emotion]['false_pos'])

                # predict_pos = thresh_dict[thresh][emotion]['true_pos'] + thresh_dict[thresh][emotion]['false_pos']
                # prev_predict_pos = thresh_dict[prev_thresh][emotion]['true_pos'] + thresh_dict[prev_thresh][emotion]['false_pos']
                # if predict_pos < prev_predict_pos:
                #     print('smaller')
                # else:
                #     print('geq')



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
        for i,_ in enumerate(Pool(1).imap(f, threshes, chunksize=10)):
            while not out_q.empty():
                thresh_dict.update(out_q.get())
            bar.update(i)
        json.dump(thresh_dict, open(thresh_file, 'w'))
    validate_thresh_dict(thresh_dict)

    for emotion in all_emotions:
        x_vals = [thresh_dict[thresh][emotion]['false_pos']/thresh_dict[thresh][emotion]['total_neg'] for thresh in sorted(thresh_dict.keys()) if emotion in thresh_dict[thresh] and thresh_dict[thresh][emotion]['total_neg'] != 0]
        y_vals = [thresh_dict[thresh][emotion]['true_pos']/thresh_dict[thresh][emotion]['total_pos'] for thresh in sorted(thresh_dict.keys()) if emotion in thresh_dict[thresh] and thresh_dict[thresh][emotion]['total_pos']  != 0]
        if x_vals and y_vals:
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
                total_pos = thresh_dict[thresh][emotion]['total_pos']
                false_pos = thresh_dict[thresh][emotion]['false_pos']
                true_pos = thresh_dict[thresh][emotion]['true_pos']
                if total_pos and (false_pos + true_pos):
                    precision = true_pos/(false_pos + true_pos)
                    recall = true_pos/total_pos
                    out_vals[thresh] = [precision, recall]
        x_vals = [out_vals[thresh][0] for thresh in sorted(out_vals.keys())]
        y_vals = [out_vals[thresh][1] for thresh in sorted(out_vals.keys())]
        z_vals = [float(x) for x in sorted(out_vals.keys())]

        x_vals = [thresh_dict[thresh][emotion]['true_pos']/(thresh_dict[thresh][emotion]['false_pos'] + thresh_dict[thresh][emotion]['true_pos']) for thresh in sorted(thresh_dict.keys()) if emotion in thresh_dict[thresh] and (thresh_dict[thresh][emotion]['false_pos'] + thresh_dict[thresh][emotion]['true_pos']) != 0]
        y_vals = [thresh_dict[thresh][emotion]['true_pos']/(thresh_dict[thresh][emotion]['total_pos']) for thresh in sorted(thresh_dict.keys()) if emotion in thresh_dict[thresh] and thresh_dict[thresh][emotion]['total_pos']  != 0]
        if x_vals and y_vals:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(x_vals, y_vals)
            ax.set_xlabel('Precision')
            ax.set_ylabel('Recall')
            plt.savefig('{0}_pr'.format(emotion))
            plt.close()

        # if x_vals and y_vals and z_vals:
        #     fig = plt.figure()
        #     ax = fig.gca(projection='3d')
        #     ax.plot(x_vals, z_vals, y_vals)
        #     ax.set_xlabel('Precision')
        #     ax.set_ylabel('Threshold')
        #     ax.set_zlabel('Recall')
        #     ax.set_title(emotion + '_pr')

    plt.show()
