"""
..module:: MultiPrevalenceScorer
    :synopsis: Calculates the average prevalence score per emotion and the accuracy relative to
        ground truth labeling for a series of videos.

"""
import csv
import functools
import glob
import json
import multiprocessing
import os
import sys
from collections import defaultdict
from os.path import join

import progressbar
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append('/home/gvelchuru/')

from OpenFaceScripts import AUGui
from OpenFaceScripts.scoring.AUScorer import make_frame_emotions
from OpenFaceScripts.scoring import AUScorer
from OpenFaceScripts.helpers.SecondRunHelper import process_eyebrows, get_vid_from_dir
from OpenFaceScripts.runners import VidCropper


def find_scores(out_q, eyebrow_dict, patient_dir):
    """
    Finds the scores for a specific patient directory
    :param out_q: Queue to output results to (for multiprocessing)
    :param patient_dir: Directory to look in
    """
    patient_dir_scores = {patient_dir: defaultdict()}
    try:
        if patient_dir in eyebrow_dict['Eyebrows']:
            include_eyebrows = True
        else:
            include_eyebrows = False
        all_dict_file = join(patient_dir, 'all_dict.txt')
        if os.path.exists(all_dict_file):
            patient_emotions = make_frame_emotions(json.load(open(all_dict_file)))
        else:
            patient_emotions = AUScorer.AUScorer(patient_dir, include_eyebrows=include_eyebrows).emotions
        csv_path = join(patient_dir, os.path.basename(patient_dir).replace('_cropped', '') + '_emotions.csv')
        num_frames = int(VidCropper.duration(get_vid_from_dir(patient_dir)) * 30)

        if os.path.exists(csv_path):
            csv_dict = AUGui.csv_emotion_reader(csv_path)
            if csv_dict:
                annotated_ratio = int(num_frames / len(csv_dict))
                if annotated_ratio == 0:
                    annotated_ratio = 1
                csv_dict = {i * annotated_ratio: c for i, c in csv_dict.items()}
                for i in [x for x in csv_dict.keys() if 'None' not in csv_dict[x]]:
                    if i in patient_emotions:
                        emotionDict = patient_emotions[i]
                        if emotionDict:
                            reverse_emotions = AUScorer.reverse_emotions(emotionDict)
                            max_value = max(reverse_emotions.keys())
                            max_emotions = {'Max': reverse_emotions[max_value]}
                            prevalence_score = AUGui.prevalence_score(emotionDict)
                            if 1 < len(reverse_emotions.keys()):
                                second_prediction = reverse_emotions[sorted(reverse_emotions.keys(), reverse=True)[1]]
                                max_emotions['Second'] = second_prediction
                            to_write = csv_dict[i]
                            if to_write == 'Surprised':
                                to_write = 'Surprise'
                            elif to_write == 'Disgusted':
                                to_write = 'Disgust'
                            elif to_write == 'Afraid':
                                to_write = 'Fear'
                            patient_dir_scores[patient_dir][i] = [max_emotions, prevalence_score, to_write]
                        else:
                            patient_dir_scores[patient_dir][i] = None
                    else:
                        patient_dir_scores[patient_dir][i] = None
        out_q.put(patient_dir_scores)
    except FileNotFoundError as e:
        print(e)
        pass


def write_to_log(log):
    csv_writer = csv.writer(log)
    agreeing_scores = {}
    non_agreeing_scores = []
    for crop_dir in (x for x in out_scores if out_scores[x]):
        num_agree = 0
        secondary_agree = 0
        num_blanks = 0
        for frame in (frame for frame in out_scores[crop_dir] if frame):
            score_list = out_scores[crop_dir][frame]

            if score_list[0]:
                emotionStrings = score_list[0]
                prevalence_score = score_list[1]
                annotated_string = score_list[2]
                if annotated_string == '':
                    num_blanks += 1
                else:
                    if annotated_string in emotionStrings['Max']:
                        num_agree += 1
                    elif prevalence_score < 1.5 and (score_list[2] in ['Neutral', 'Sleeping']):
                        num_agree += 1
                    elif 'Second' in emotionStrings and annotated_string in emotionStrings['Second']:
                        secondary_agree += 1
                    else:
                        non_agreeing_scores.append(score_list)
        agreeing_scores[crop_dir] = [num_agree, secondary_agree, len(out_scores[crop_dir])]
        csv_writer.writerow([crop_dir] + agreeing_scores[crop_dir] + [len(scores[crop_dir]) - num_blanks])
    for vid_score in sorted(non_agreeing_scores, key=lambda score: score[0]['Max']):
        vid_score = [vid_score[0]['Max']] + [vid_score[1]] + [vid_score[2]]
        csv_writer.writerow(vid_score)
        print(vid_score)


if __name__ == '__main__':
    OpenDir = sys.argv[sys.argv.index('-d') + 1]
    os.chdir(OpenDir)
    patient_dirs = glob.glob('*cropped')  # Directories have been previously cropped by CropAndOpenFace
    scores = defaultdict()
    out_scores = defaultdict()
    scores_file = 'scores.txt'
    if os.path.exists(scores_file):
        scores = json.load(open(scores_file))
    original_len = len(scores)
    remaining = [x for x in patient_dirs if x not in scores]
    if len(remaining) > 0:
        out_q = multiprocessing.Manager().Queue()
        eyebrow_dict = process_eyebrows(OpenDir, open(join(OpenDir, 'eyebrows.txt')))
        f = functools.partial(find_scores, out_q, eyebrow_dict)
        bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(remaining))
        for i, _ in enumerate(Pool().imap(f, remaining), 1):
            bar.update(i)
        while not out_q.empty():
            scores.update(out_q.get())
        if len(scores) != original_len:
            json.dump(scores, open(scores_file, 'w'))
    for patient_dir in (x for x in scores if scores[x]):
        # Maps frame to list if frame has been detected by OpenFace and it has been annotated
        out_scores[patient_dir] = {frame: score_list for frame, score_list in scores[patient_dir].items() if
                                   score_list}

    with open(join(OpenDir, 'OpenFaceScores.csv'), 'w') as log:
        write_to_log(log)
