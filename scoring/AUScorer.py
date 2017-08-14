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
from OpenFaceScripts.scoring import OpenFaceScorer


def emotion_templates(include_similar):
    emotion_templates = {
        'Angry': [[23, 7, 17, 4, 2]],
        'Fear': [[20, 4, 1, 5, 7]],
        'Sad': [[15, 1, 4, 17, 10]],
        'Happy': [[12, 6, 26, 10, 23]],
        'Surprise': [[27, 2, 1, 5, 26]],
        'Disgust': [[9, 7, 4, 17, 6]]
    }
    if include_similar:
        similar_arr = [
            [12, 20, 23, 15]
        ]
        for emotion, au_list_arr in emotion_templates.items():
            for similar in similar_arr:
                for num in similar:
                    if num in au_list_arr[0]:
                        for other_num in [x for x in similar if x is not num]:
                            au_list_arr.append(replace(au_list_arr[0], num, other_num))
    for emotion in emotion_templates:
        emotion_templates[emotion] = [sorted(v) for v in emotion_templates[emotion]]

    return emotion_templates


class AUScorer:
    """
    Main scorer
    """

    def __init__(self, dir, au_thresh=0, include_eyebrows=True):
        """
        Default constructor.

        :param dir: Directory with au files to score
        :type dir: str
        :param au_thresh: Minimum threshold (0-5) for considering AUs to be present
        :type au_thresh: float
        :param include_eyebrows: Whether eyebrows should be considered
        :type include_eyebrows: bool
        """
        self.include_eyebrows = include_eyebrows
        self.include_similar = False
        original_dir = os.getcwd()
        os.chdir(dir)
        au_file = 'au.txt'  # Replace with name of action units file
        if not os.path.exists('au.txt'):
            os.chdir(original_dir)
            raise FileNotFoundError("{0} does not exist!".format(os.path.join(dir, au_file)))
        open_face_arr, open_face_dict = OpenFaceScorer.OpenFaceScorer.make_au_parts(au_file)

        # Creates a dictionary mapping each frame in the video to a dictionary containing the frame's action units
        # and their amounts
        au_dict = {frame: {label: open_face_arr[frame][num] for label, num in open_face_dict.items()
                    if 'AU' in label} for frame in range(len(open_face_arr))}
        self.x_y_dict = {frame: {label: open_face_arr[frame][num] for label, num in open_face_dict.items()
                    if 'x_' in label or 'y_' in label} for frame in range(len(open_face_arr))}
        self.x_y_dict = {frame: frame_dict for frame, frame_dict in self.x_y_dict.items() if any(frame_dict.values())}
        self.presence_dict = defaultdict()
        for frame in range(len(open_face_arr)):
            if open_face_arr[frame][open_face_dict['success']]:
                self.presence_dict[frame] = defaultdict()
                curr_frame = au_dict[frame]
                for label in curr_frame:
                    if 'c' in label and curr_frame[label] == 1 and not self.is_eyebrow(label):
                        r_label = label.replace('c', 'r')
                        if (r_label in curr_frame and curr_frame[r_label] >= au_thresh) or (
                                    r_label not in curr_frame):
                            self.presence_dict[frame][label] = curr_frame[label]
                            if r_label in curr_frame:
                                self.presence_dict[frame][r_label] = curr_frame[r_label]

        frame_emotions = self.make_frame_emotions(self.presence_dict)
        self.emotions = {frame: frame_dict for frame, frame_dict in frame_emotions.items()}
        os.chdir(original_dir)

    def is_eyebrow(self, label):
        if self.include_eyebrows:
            return False
        elif self.return_num(label) in [1, 2, 4]:
            return True

    def get_emotions(self, index):
        """
        Gets the emotions at a specific index.

        :param index:  Index to look for emotions at
        :return: Emotions at index if index is in the list of emotion indices, else None
        """
        return self.emotions[index] if index in self.emotions else None

    def make_frame_emotions(self, presence_dict):
        frame_emotion_dict = {
            frame: self.find_all_lcs(sorted([self.return_num(au) for au in au_dict if 'c' in au]))
            for frame, au_dict in presence_dict.items()}

        for frame, emotion_dict in frame_emotion_dict.items():
            for emotion in emotion_dict:
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
        emote_template = emotion_templates(self.include_similar)
        return {
            emotion: [back_track(LCS(template, aus), template, aus, len(template), len(aus)) for template in
                      template_arr]
            for emotion, template_arr in emote_template.items()}

    def convert_aus_to_scores(self, arr, frame, presence_dict):
        frame_presence = presence_dict[frame]
        scores = []
        for au in frame_presence:
            if 'c' in au and self.return_num(au) in arr:
                r_label = au.replace('c', 'r')
                if r_label in frame_presence:
                    scores.append(frame_presence[r_label] / 5)  # divide by 5 to normalize
                else:
                    scores.append(1.0 / 5)  # divide by 5 to normalize
        return np.sum(scores)

    @staticmethod
    def return_num(string):
        return int(re.findall("\d+", string)[0])


def emotion_list():
    """
    Create standard emotion list.

    :return: List with the emotions Angry, Fear, Sad, Happy, Surprise, and Disgust.
    """
    return ['Angry', 'Fear', 'Sad', 'Happy', 'Surprise', 'Disgust']


def replace(arr, num, other_num):
    """
    Remove an element from an array and add another one.
    :param arr: Array.
    :param num: Element to remove.
    :param other_num: Element to add.
    :return: Changed array.
    """
    small_arr = [x for x in arr if x is not num]
    small_arr.append(other_num)
    return small_arr


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


def reverse_emotions(emotionDict):
    """
    Creates a dictionary mapping between the values of emotions and all the emotions with that value.

    :param emotionDict: Mapping between emotions and their scores
    :type emotionDict: dict
    :return: Reverse mapped dictionary
    """
    return {value: [x for x in emotionDict if emotionDict[x] == value] for value in
            emotionDict.values()}


if __name__ == '__main__':
    directory = sys.argv[sys.argv.index('-d') + 1]
    score = AUScorer(directory)
