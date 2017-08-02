"""
..module:: MultiPrevalenceScorer
    :synopsis: Calculates the average prevalence score per emotion and the accuracy relative to
        ground truth labeling for a series of videos.

"""
import csv
import json
import multiprocessing
import os
import sys
import glob
from collections import defaultdict

from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import AUScorer, AUGui, VidCropper


class MultiPrevalenceScorer:
    """
    Main scoring class.

    """

    def __init__(self, OpenDir):
        os.chdir(OpenDir)
        patient_dirs = glob.glob('*cropped')  # Directories have been previously cropped by CropAndOpenFace
        scores = defaultdict()
        out_scores = defaultdict()
        scores_file = 'scores.txt'
        if os.path.exists(scores_file):
            scores = json.load(open(scores_file))
        original_len = len(scores)
        remaining = [x for x in patient_dirs if x not in scores or not scores[x]]
        if len(remaining) > 0:
            patient_dirs.sort()
            self.out_q = multiprocessing.Manager().Queue()
            Pool().map(self.find_scores, remaining)
            while not self.out_q.empty():
                scores.update(self.out_q.get())
            if len(scores) != original_len:
                json.dump(scores, open(scores_file, 'w'))
        for patient_dir in (x for x in scores if scores[x]):
            # Maps frame to list if frame has been detected by OpenFace and it has been annotated
            out_scores[patient_dir] = {frame: list for frame, list in
                                       ((f, list) for f, list in scores[patient_dir].items() if list and
                                        (list[2] is not None and list[2] != 'N/A'))}

        with open(os.path.join(OpenDir, 'OpenFaceScores.csv'), 'w') as log:
            csv_writer = csv.writer(log)
            agreeing_scores = {}
            non_agreeing_scores = []
            for crop_dir in (x for x in out_scores if out_scores[x]):
                num_agree = 0
                secondary_agree = 0
                num_blanks = 0
                for frame in (frame for frame in out_scores[crop_dir] if frame):
                    score_list = out_scores[crop_dir][frame]
                    if score_list[2] == 'Surprised':
                        score_list[2] = 'Surprise'
                    elif score_list[2] == 'Disgusted':
                        score_list[2] = 'Disgust'
                    elif score_list[2] == 'Afraid':
                        score_list[2] = 'Fear'

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
            for score in sorted(non_agreeing_scores, key=lambda score: score[0]['Max']):
                score = [score[0]['Max']] + [score[1]] + [score[2]]
                csv_writer.writerow(score)
                print(score)

    def find_scores(self, patient_dir):
        """
        Finds the scores for a specific patient directory

        :param patient_dir: Directory to look in
        """
        print(patient_dir)
        patient_dir_scores = {patient_dir: defaultdict()}
        try:
            all_dict_file = os.path.join(patient_dir, 'all_dict.txt')
            if os.path.exists(all_dict_file):
                patient_emotions = {int(k): v for k, v in json.load(open(all_dict_file)).items()}
            else:
                patient_emotions = AUScorer.AUScorer(patient_dir).emotions
            csv_paths = glob.glob(os.path.join(patient_dir, '*.csv'))
            csv_dict = None
            num_frames = VidCropper.duration(os.path.join(patient_dir, 'out.avi')) * 30
            num_frames = int(num_frames)
            if len(csv_paths) == 1:
                csv_dict = AUGui.csv_emotion_reader(csv_paths[0])
                if csv_dict:
                    annotated_ratio = int(num_frames / len(csv_dict))
                    if annotated_ratio == 0:
                        annotated_ratio = 1
                    csv_dict = {i * annotated_ratio: c for i, c in csv_dict.items()}
            if csv_dict:
                for i in csv_dict.keys():
                    if i in patient_emotions:
                        emotionDict = patient_emotions[i]
                        if emotionDict:
                            reverse_emotions = AUScorer.reverse_emotions(emotionDict)
                            max_value = max(reverse_emotions.keys())
                            max_emotions = {'Max' : reverse_emotions[max_value][0]}
                            prevalence_score = AUGui.prevalence_score(emotionDict)
                            if 1 < len(reverse_emotions.keys()):
                                second_prediction = reverse_emotions[sorted(reverse_emotions.keys(), reverse=True)[1]]
                                max_emotions['Second'] = second_prediction
                            patient_dir_scores[patient_dir][i] = [max_emotions, prevalence_score]
                            if i in csv_dict:
                                patient_dir_scores[patient_dir][i].append(csv_dict[i])
                            else:
                                patient_dir_scores[patient_dir][i].append('N/A')
                        else:
                            patient_dir_scores[patient_dir][i] = None
                    else:
                        patient_dir_scores[patient_dir][i] = None
            self.out_q.put(patient_dir_scores)
        except FileNotFoundError as e:
            print(e)
            pass


if __name__ == '__main__':
    dir = sys.argv[sys.argv.index('-d') + 1]
    scorer = MultiPrevalenceScorer(dir)
