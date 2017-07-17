"""
..module:: MultiPrevalenceScorer
    :synopsis: Calculates the average prevalence score per emotion and the accuracy relative to
        ground truth labeling for a series of videos.

"""
import os
import sys
import glob
from collections import defaultdict

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import AUScorer, AUGui


class MultiPrevalenceScorer:
    """
    Main scoring class.

    """

    def __init__(self, dir):
        os.chdir(dir)
        patient_dirs = glob.glob('*cropped')  # Directories have been previously cropped by CropAndOpenFace
        scores = defaultdict()
        for patient_dir in sorted(patient_dirs):
            scores[patient_dir] = defaultdict()
            try:
                scorer = AUScorer.AUScorer(patient_dir)
                frames = glob.glob(os.path.join(patient_dir, '*.png'))
                csv_paths = glob.glob(os.path.join(patient_dir, '*coordinates.csv'))
                csv_dict = None
                if len(csv_paths) == 1:
                    csv_dict = AUGui.csv_emotion_reader(csv_paths[0])
                    annotated_ratio = int(len(frames)/len(csv_dict.keys()))
                    csv_dict = {i * annotated_ratio: c for i, c in csv_dict.items()}
                for i in range(len(frames)):
                    emotionDict = scorer.get_emotions(i)
                    if emotionDict:
                        reverse_emotions = AUScorer.reverse_emotions(emotionDict)
                        max_value = max(reverse_emotions.keys())
                        if len(reverse_emotions[max_value]) > 1:
                            max_emotion = None
                        else:
                            max_emotion = reverse_emotions[max_value][0]
                        prevalence_score = AUGui.prevalence_score(emotionDict)
                        scores[patient_dir][i] = [max_emotion, prevalence_score]
                        if csv_dict:
                            if i in csv_dict.keys():
                                scores[patient_dir][i].append(csv_dict[i])
                            else:
                                scores[patient_dir][i].append('N/A')
                        else:
                            scores[patient_dir][i].append(None)
                    else:
                        scores[patient_dir][i] = None
            except FileNotFoundError as e:
                print(e)
                continue
        pass



if __name__ == '__main__':
    dir = sys.argv[sys.argv.index('-d') + 1]
    scorer = MultiPrevalenceScorer(dir)
