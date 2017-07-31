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
        remaining = [x for x in patient_dirs if x not in scores]
        if len(remaining) > 0:
            patient_dirs.sort()
            self.out_q = multiprocessing.Manager().Queue()
            Pool().map(self.find_scores, remaining)
            while not self.out_q.empty():
                scores.update(self.out_q.get())
            json.dump(scores, open(scores_file, 'w'))
        for patient_dir in scores:
            if scores[patient_dir]:
                out_scores[patient_dir] = {frame: list for frame, list in
                                           ((f, list) for f, list in scores[patient_dir].items() if
                                            (list[2] and list[2] != 'N/A'))}

        with open(os.path.join(OpenDir, 'OpenFaceScores.csv'), 'w') as log:
            csv_writer = csv.writer(log)
            agreeing_scores = {}
            non_agreeing_scores = []
            for crop_dir in out_scores:
                if out_scores[crop_dir]:
                    num_agree = 0
                    for frame in out_scores[crop_dir]:
                        score_list = out_scores[crop_dir][frame]
                        if score_list[2] == 'Surprised':
                            score_list[2] = 'Surprise'
                        elif score_list[2] == 'Disgusted':
                            score_list[2] = 'Disgust'
                        elif score_list[2] == 'Afraid':
                            score_list[2] = 'Fear'

                        if score_list[0]:
                            emotionString = score_list[0]
                            prevalence_score = score_list[1]
                            annotated_string = score_list[2]
                            if prevalence_score < 1.5:
                                if score_list[2] == 'Neutral' or score_list[2] == 'Sleeping' or emotionString == annotated_string:
                                    num_agree += 1
                                else:
                                    non_agreeing_scores.append(score_list)
                            else:
                                if emotionString == annotated_string:
                                    num_agree += 1
                                else:
                                    non_agreeing_scores.append(score_list)
                    agreeing_scores[crop_dir] = [num_agree, len(out_scores[crop_dir])]
                    csv_writer.writerow([crop_dir] + agreeing_scores[crop_dir] + [int(VidCropper.duration(os.path.join(crop_dir, 'out.avi')) * 30)])
            for score in sorted(non_agreeing_scores, key=lambda score: score[0]):
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
            num_frames = int(VidCropper.duration(os.path.join(patient_dir, 'out.avi')) * 30)
            if len(csv_paths) == 1:
                csv_dict = AUGui.csv_emotion_reader(csv_paths[0])
                if csv_dict:
                    annotated_ratio = int(num_frames / len(csv_dict))
                    if annotated_ratio == 0:
                        annotated_ratio = 1
                    csv_dict = {i * annotated_ratio: c for i, c in csv_dict.items()}
            for i in (x for x in range(num_frames) if x in patient_emotions):
                emotionDict = patient_emotions[i]
                if emotionDict:
                    reverse_emotions = AUScorer.reverse_emotions(emotionDict)
                    max_value = max(reverse_emotions.keys())
                    if len(reverse_emotions[max_value]) > 1:
                        max_emotion = None
                    else:
                        max_emotion = reverse_emotions[max_value][0]
                    prevalence_score = AUGui.prevalence_score(emotionDict)
                    patient_dir_scores[patient_dir][i] = [max_emotion, prevalence_score]
                    if csv_dict:
                        if i in csv_dict:
                            patient_dir_scores[patient_dir][i].append(csv_dict[i])
                        else:
                            patient_dir_scores[patient_dir][i].append('N/A')
                    else:
                        patient_dir_scores[patient_dir][i].append(None)
            self.out_q.put(patient_dir_scores)
        except FileNotFoundError as e:
            print(e)
            pass


if __name__ == '__main__':
    dir = sys.argv[sys.argv.index('-d') + 1]
    scorer = MultiPrevalenceScorer(dir)
