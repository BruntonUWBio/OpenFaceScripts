"""
.. module:: AUScorer
    :synopsis: An action unit scorer
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from OpenFaceScripts.scoring import OpenFaceScorer

AUList = [
    '1', '2', '4', '5', '6', '7', '9', '10', '12', '14', '15', '17', '20', '23', '25', '26', '28', '45',
          'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
          'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'confidence']
LandmarkList = [
    x for x in AUList if 'gaze' not in x and 'pose' not in x and 'confidence' not in x]
TrainList = [x for x in AUList if 'confidence' not in x]
include_similar = False


def emotion_templates(include_similar: bool) -> dict:
    emotion_templates = {
        'Angry': [list(map(str, [23, 7, 17, 4, 2]))],
        'Fear': [list(map(str, [20, 4, 1, 5, 7]))],
        'Sad': [list(map(str, [15, 1, 4, 17, 10]))],
        'Happy': [list(map(str, [12, 6, 26, 10, 23]))],
        'Surprise': [list(map(str, [27, 2, 1, 5, 26]))],
        'Disgust': [list(map(str, [9, 7, 4, 17, 6]))]
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
                            au_list_arr.append(
                                replace(au_list_arr[0], num, other_num))

    for emotion in emotion_templates:
        emotion_templates[emotion] = [sorted(v)
                                      for v in emotion_templates[emotion]]

    return emotion_templates


def convert_dict_to_int(dict):
    return {int(k): v for k, v in dict.items()}


class AUScorer:

    """
    Main scorer
    """

    def __init__(self, dir, include_eyebrows=True):
        """
        Default constructor.

        :param dir: Directory with au files to score
        :type dir: str
        :param include_eyebrows: Whether eyebrows should be considered
        :type include_eyebrows: bool
        """
        self.include_eyebrows = include_eyebrows
        original_dir = os.getcwd()
        os.chdir(dir)
        au_file = 'au.txt'  # Replace with name of action units file

        if not os.path.exists('au.txt'):
            os.chdir(original_dir)
            raise FileNotFoundError(
                "{0} does not exist!".format(os.path.join(dir, au_file)))
        open_face_arr, open_face_dict = OpenFaceScorer.OpenFaceScorer.make_au_parts(
            au_file)

        # Creates a dictionary mapping each frame in the video to a dictionary containing the frame's action units
        # and their amounts
        au_dict = {frame: {label: open_face_arr[frame][num] for label, num in open_face_dict.items()
                           if any(x in label for x in ['gaze', 'pose_R', 'AU', 'confidence'])} for frame in
                   range(len(open_face_arr))}
        self.x_y_dict = {frame: {label: open_face_arr[frame][num] for label, num in open_face_dict.items()
                                 if 'x_' in label or 'y_' in label} for frame in range(len(open_face_arr))}
        self.x_y_dict = {frame: frame_dict for frame,
                         frame_dict in self.x_y_dict.items() if any(frame_dict.values())}
        self.presence_dict = defaultdict()

        for frame in range(len(open_face_arr)):
            if open_face_arr[frame][open_face_dict['success']]:
                self.presence_dict[frame] = defaultdict()
                curr_frame = au_dict[frame]

                for label in curr_frame:
                    if 'c' in label and curr_frame[label] == 1 and not self.is_eyebrow(label):
                        r_label = label.replace('c', 'r')

                        if r_label not in curr_frame:
                            stripped_label = str(return_num(label))
                            self.presence_dict[frame][
                                stripped_label] = str(curr_frame[label])
                        else:
                            stripped_r_label = str(return_num(r_label))
                            self.presence_dict[frame][
                                stripped_r_label] = str(curr_frame[r_label])
                    elif 'r' in label and not self.is_eyebrow(label):
                        c_label = label.replace('r', 'c')

                        if c_label not in curr_frame:
                            self.presence_dict[frame][
                                str(return_num(label))] = str(curr_frame[label])
                    elif 'pose_R' in label or 'gaze' in label or 'confidence' in label:
                        self.presence_dict[frame][label] = curr_frame[label]

        self.emotions = make_frame_emotions(self.presence_dict)
        os.chdir(original_dir)

    def is_eyebrow(self, label: str) -> bool:
        if self.include_eyebrows:
            return False
        elif return_num(label) in [1, 2, 4]:
            return True

    def get_emotions(self, index: int):
        """
        Gets the emotions at a specific index.

        :param index:  Index to look for emotions at
        :return: Emotions at index if index is in the list of emotion indices, else None
        """

        return self.emotions[index] if index in self.emotions else None


def make_frame_emotions(presence_dict: dict) -> dict:
    frame_emotion_dict = {
        frame: find_all_lcs(
            sorted([au for au in au_dict if 'pose' not in au and 'gaze' not in au]))

        for frame, au_dict in presence_dict.items()}

    for frame, emotion_dict in frame_emotion_dict.items():
        for emotion in emotion_dict:
            emotion_dict[emotion] = [
                x for x in emotion_dict[emotion] if x]  # remove empties

            for index, arr in enumerate(emotion_dict[emotion]):
                emotion_dict[emotion][index] = convert_aus_to_scores(
                    arr, frame, presence_dict)

            if len(emotion_dict[emotion]):
                emotion_dict[emotion] = max([x for x in emotion_dict[emotion]])
            else:
                emotion_dict[emotion] = None
        frame_emotion_dict[frame] = {
            k: v for k, v in emotion_dict.items() if v}

    return frame_emotion_dict


def find_all_lcs(aus: list) -> dict:
    emote_template = emotion_templates(include_similar)

    return {
        emotion: [back_track(LCS(template, aus), template, aus, len(template), len(aus)) for template in
                  template_arr]

        for emotion, template_arr in emote_template.items()}


def convert_aus_to_scores(arr: list, frame: int, presence_dict: dict) -> np.ndarray:
    frame_presence = presence_dict[frame]
    scores = []

    for au, val in ((x, val) for x, val in frame_presence.items() if x in arr):
        scores.append(val)
    scores = np.array(scores, np.float64)

    return np.sum(scores)


def return_num(string: str) -> int:
    return int(re.findall("\d+", string)[0])


def emotion_list() -> list:
    """
    Create standard emotion list.

    :return: List with the emotions Angry, Fear, Sad, Happy, Surprise, and Disgust.
    """

    return ['Angry', 'Fear', 'Sad', 'Happy', 'Surprise', 'Disgust']


def replace(arr: list, num: int, other_num: int):
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
