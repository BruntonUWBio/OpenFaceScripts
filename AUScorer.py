"""
.. module:: AUScorer
    :synopsis: An action unit scorer
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import OpenFaceScorer


class AUScorer:
    """
    Main scorer
    """

    def __init__(self, dir, au_thresh=0, include_eyebrows=True):
        """
        Default constructor.

        :param dir: Directory with au files to score
        :param au_thresh: Minimum threshold (0-5) for considering AUs to be present
        :param include_eyebrows: Whether eyebrows should be considered
        """
        self.include_eyebrows = include_eyebrows
        self.include_similar = False
        os.chdir(dir)
        au_file = 'au.txt'  # Replace with name of action units file
        open_face_arr, open_face_dict = OpenFaceScorer.OpenFaceScorer.make_au_parts(au_file)

        # Creates a dictionary mapping each frame in the video to a dictionary containing the frame's action units
        # and their amounts
        au_dict = {
            frame: {label: open_face_arr[frame][num] for label, num in open_face_dict.items()
                    if 'AU' in label} for frame in range(len(open_face_arr))}
        self.presence_dict = defaultdict()
        for frame in range(len(open_face_arr)):
            self.presence_dict[frame] = defaultdict()
            curr_frame = au_dict[frame]
            curr_frame_keys = curr_frame.keys()
            for label in curr_frame_keys:
                if 'c' in label and curr_frame[label] == 1 and not self.is_eyebrow(label):
                    r_label = label.replace('c', 'r')
                    if (r_label in curr_frame_keys and curr_frame[r_label] >= au_thresh) or (
                                r_label not in curr_frame_keys):
                        self.presence_dict[frame][label] = curr_frame[label]
                        if r_label in curr_frame_keys:
                            self.presence_dict[frame][r_label] = curr_frame[r_label]

        frame_emotions = self.make_frame_emotions(self.presence_dict)
        self.emotions = {frame: frame_dict for frame, frame_dict in frame_emotions.items()}

    @staticmethod
    def emotion_list():
        """
        Create standard emotion list.

        :return: List with the emotions Angry, Fear, Sad, Happy, Surprise, and Disgust.
        """
        return ['Angry', 'Fear', 'Sad', 'Happy', 'Surprise', 'Disgust']

    def is_eyebrow(self, label):
        if self.include_eyebrows:
            return False
        elif self.return_num(label) in [1, 2, 4]:
            return True

    def emotion_templates(self):
        emotion_templates = {
            'Angry': [[23, 7, 17, 4, 2]],
            'Fear': [[20, 4, 1, 5, 7]],
            'Sad': [[15, 1, 4, 17, 10]],
            'Happy': [[12, 6, 26, 10, 23]],
            'Surprise': [[27, 2, 1, 5, 26]],
            'Disgust': [[9, 7, 4, 17, 6]]
        }
        if self.include_similar:
            similar_arr = [
                [12, 20, 23, 15]
            ]
            for emotion, au_list_arr in emotion_templates.items():
                for similar in similar_arr:
                    for num in similar:
                        if num in au_list_arr[0]:
                            for other_num in [x for x in similar if x is not num]:
                                au_list_arr.append(self.replace(au_list_arr[0], num, other_num))
        for emotion in emotion_templates.keys():
            emotion_templates[emotion] = [sorted(v) for v in emotion_templates[emotion]]

        return emotion_templates

    @staticmethod
    def replace(arr, num, other_num):
        small_arr = [x for x in arr if x is not num]
        small_arr.append(other_num)
        large_set = set(small_arr)
        return list(large_set)

    def get_emotions(self, index):
        return self.emotions[index] if index in self.emotions else None

    def make_frame_emotions(self, presence_dict):
        frame_emotion_dict = {
            frame: self.find_all_lcs(sorted([self.return_num(au) for au in au_dict.keys() if 'c' in au]))
            for frame, au_dict in presence_dict.items()}

        for frame, emotion_dict in frame_emotion_dict.items():
            for emotion in emotion_dict.keys():
                emotion_dict[emotion] = [x for x in emotion_dict[emotion] if x]  # remove empties
                for index, arr in enumerate(emotion_dict[emotion]):
                    emotion_dict[emotion][index] = self.convert_aus_to_scores(arr, frame, presence_dict)
                if len(emotion_dict[emotion]):
                    emotion_dict[emotion] = max([x for x in emotion_dict[emotion]])
                else:
                    emotion_dict[emotion] = None
            frame_emotion_dict[frame] = {k: v for k, v in emotion_dict.items() if v}

        return frame_emotion_dict

    def find_all_lcs(self, aus):
        emote_template = self.emotion_templates()
        return {
            emotion: [back_track(LCS(template, aus), template, aus, len(template), len(aus)) for template in
                      template_arr]
            for emotion, template_arr in emote_template.items()}

    def convert_aus_to_scores(self, arr, frame, presence_dict):
        frame_presence = presence_dict[frame]
        scores = []
        for au in frame_presence.keys():
            if 'c' in au and self.return_num(au) in arr:
                r_label = au.replace('c', 'r')
                if r_label in frame_presence.keys():
                    scores.append(frame_presence[r_label] / 5)  # divide by 5 to normalize
                else:
                    scores.append(1.0 / 5)  # divide by 5 to normalizes
        return np.sum(scores)

    @staticmethod
    def return_num(string):
        return int(re.findall("\d+", string)[0])


def back_track(C, X, Y, i, j):
    if i == 0 or j == 0:
        return []
    elif X[i - 1] == Y[j - 1]:
        return back_track(C, X, Y, i - 1, j - 1) + [X[i - 1]]
    else:
        if C[i][j - 1] > C[i - 1][j]:
            return back_track(C, X, Y, i, j - 1)
        else:
            return back_track(C, X, Y, i - 1, j)


def LCS(X, Y):
    m = len(X)
    n = len(Y)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])
    return C


if __name__ == '__main__':
    directory = sys.argv[sys.argv.index('-d') + 1]
    score = AUScorer(directory)
