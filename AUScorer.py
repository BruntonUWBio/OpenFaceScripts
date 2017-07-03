import os
import re
import sys
from collections import defaultdict

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import OpenFaceScorer


class AUScorer:
    def __init__(self, dir, au_thresh=0, include_eyebrows=True):
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
        # self.exact_emotions = {frame: frame_dict for frame, frame_dict in frame_emotions.items() if
        #                       frame_dict['Exact Match']}
        self.emotions = {frame: frame_dict for frame, frame_dict in frame_emotions.items() if
                         not all(v is 0 for v in frame_dict.values())}

    @staticmethod
    def emotion_list():
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
            self.similar_arr = [
                [12, 20, 23, 15]
            ]
            for emotion, au_list_arr in emotion_templates.items():
                for similar in self.similar_arr:
                    for num in similar:
                        if num in au_list_arr[0]:
                            for other_num in [x for x in similar if x is not num]:
                                au_list_arr.append(self.replace(au_list_arr[0], num, other_num))
        for emotion in emotion_templates.keys():
            emotion_templates[emotion] = [sorted(v) for v in emotion_templates[emotion]]

        return emotion_templates

    def replace(self, arr, num, other_num):
        small_arr = [x for x in arr if x is not num]
        small_arr.append(other_num)
        large_set = set(small_arr)
        return list(large_set)

    def make_frame_emotions(self, presence_dict):
        frame_emotion_dict = {
            frame: self.find_all_lcs(sorted([self.return_num(au) for au in au_dict.keys() if 'c' in au]))
            for frame, au_dict in presence_dict.items()}

        return frame_emotion_dict

    def find_all_lcs(self, aus):
        emote_template = self.emotion_templates()
        return {emotion: max([self.lcs_length(template, aus) for template in template_arr]) for emotion, template_arr in
                emote_template.items()}

    @staticmethod
    def return_num(string):
        return int(re.findall("\d+", string)[0])

    @staticmethod
    def lcs_length(a, b):
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, ca in enumerate(a, 1):
            for j, cb in enumerate(b, 1):
                table[i][j] = (
                    table[i - 1][j - 1] + 1 if ca == cb else
                    max(table[i][j - 1], table[i - 1][j]))
        return table[-1][-1]


if __name__ == '__main__':
    directory = sys.argv[sys.argv.index('-d') + 1]
    score = AUScorer(directory)
