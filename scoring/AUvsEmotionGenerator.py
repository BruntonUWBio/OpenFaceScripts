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
from runners.SecondRunOpenFace import get_vid_from_dir

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import AUGui
from OpenFaceScripts.scoring import AUScorer
from OpenFaceScripts.runners import SecondRunOpenFace, VidCropper


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
        presence_dict = AUScorer.AUScorer(patient_dir).presence_dict
        AU_presences = {}
        for frame in presence_dict:
            AU_presences[frame] = {}
            for au in presence_dict[frame]:
                if 'c' in au and presence_dict[frame][au] == 1:
                    r_au = au.replace('c', 'r')
                    num = int(au[2:4])
                    # AU_presences[frame][num] = presence_dict[frame][r_au] if r_au in presence_dict[frame] else presence_dict[frame][au]
                    AU_presences[frame][num] = presence_dict[frame][au]
            for num in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]:
                if num not in AU_presences[frame]:
                    AU_presences[frame][num] = 0


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
                    if i in AU_presences:
                        auDict = AU_presences[i]
                        if auDict:
                            to_write = csv_dict[i]
                            if to_write == 'Surprised':
                                to_write = 'Surprise'
                            elif to_write == 'Disgusted':
                                to_write = 'Disgust'
                            elif to_write == 'Afraid':
                                to_write = 'Fear'
                            patient_dir_scores[patient_dir][i] = [auDict, to_write]
                        else:
                            patient_dir_scores[patient_dir][i] = None
                    else:
                        patient_dir_scores[patient_dir][i] = None
        out_q.put(patient_dir_scores)
    except FileNotFoundError as e:
        print(e)
        pass


if __name__ == '__main__':
    OpenDir = sys.argv[sys.argv.index('-d') + 1]
    os.chdir(OpenDir)
    patient_dirs = glob.glob('*cropped')  # Directories have been previously cropped by CropAndOpenFace
    scores = defaultdict()
    scores_file = 'au_emotes.txt'
    if os.path.exists(scores_file):
        scores = json.load(open(scores_file))
    original_len = len(scores)
    remaining = [x for x in patient_dirs if x not in scores or not scores[x]]
    if len(remaining) > 0:
        patient_dirs.sort()
        out_q = multiprocessing.Manager().Queue()
        eyebrow_dict = SecondRunOpenFace.process_eyebrows(OpenDir, open(join(OpenDir, 'eyebrows.txt')))
        f = functools.partial(find_scores, out_q, eyebrow_dict)
        bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(remaining))
        for i, _ in enumerate(Pool().imap(f, remaining), 1):
            bar.update(i)
        while not out_q.empty():
            scores.update(out_q.get())
        if len(scores) != original_len:
            json.dump(scores, open(scores_file, 'w'))