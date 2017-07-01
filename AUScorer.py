import os
import re
import sys
from collections import defaultdict

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import OpenFaceScorer


class AUScorer:
    def __init__(self, dir):
        self.include_similar = False
        os.chdir(dir)
        au_file = 'au.txt'  # Replace with name of action units file
        open_face_arr, open_face_dict = OpenFaceScorer.OpenFaceScorer.make_au_parts(au_file)

        # Creates a dictionary mapping each frame in the video to a dictionary containing the frame's action units
        # and their amounts
        au_dict = {
            frame: {label: open_face_arr[frame][num] for label, num in open_face_dict.items()
                    if 'AU' in label} for frame in range(len(open_face_arr))}
        presence_dict = defaultdict()
        for frame in range(1, len(open_face_arr)):
            presence_dict[frame] = defaultdict()
            curr_frame = au_dict[frame]
            for label in curr_frame.keys():
                if 'c' in label and curr_frame[label] == 1:
                    presence_dict[frame][label] = curr_frame[label]
                    r_label = label.replace('c', 'r')
                    if r_label in curr_frame.keys():
                        presence_dict[frame][r_label] = curr_frame[r_label]
        frame_emotions = self.make_frame_emotions(presence_dict)
        # self.exact_emotions = {frame: frame_dict for frame, frame_dict in frame_emotions.items() if
        #                       frame_dict['Exact Match']}
        self.emotions = {frame: frame_dict for frame, frame_dict in frame_emotions.items() if
                         not all(v is 0 for v in frame_dict.values())}

    def emotion_dict(self, nums):
        # From CK+ database
        # all_emotion_dict = {
        #     'Angry': [23, 24],
        #     'Disgust_1': [9],
        #     'Disgust_2': [10],
        #     'Fear_1': [1, 2, 4],
        #     'Fear_2': [1, 2, 5],  # 'Unless AU5 is of intensity E then AU4 can be absent'
        #     'Happy': [12],
        #     'Sadness_1': [1, 4, 15],
        #     'Sadness_2': [11],
        #     'Sadness_3': [6, 15],
        #     'Surprise_1': [1, 2],
        #     'Surprise_2': [5],  # 'Intensity of 5 must not be stronger than B',
        #     'Contempt': [14]
        # }

        emotion_dict = self.emotion_distance(nums)
        return emotion_dict



        # for emotion, emotion_nums in all_emotion_dict.items():
        #     if emotion_nums == nums:
        #         emotion_dict['Exact Match'].append(emotion)
        #     elif set(emotion_nums).issubset(set(nums)):
        #         emotion_dict['Possible Match'].append(emotion)
        # if not emotion_dict['Possible Match']:
        #     emotion_dict['Possible Match'] = ['Neutral']  # Neutral case
        # return emotion_dict

    @staticmethod
    def make_emotion_matrix():
        emotion_matrix = {
            1: [4, 1, 1, 4, 1, 4],
            2: [4, 2, 2, 4, 1, 4],
            4: [1, 1, 1, 4, 4, 1],
            5: [4, 2, 4, 4, 1, 4],
            6: [4, 4, 4, 1, 4, 2],
            7: [4, 4, 4, 1, 4, 2],
            9: [2, 4, 4, 2, 3, 0],
            10: [2, 2, 2, 1, 2, 2],
            12: [4, 4, 4, 0, 4, 4],
            15: [3, 4, 1, 4, 4, 4],
            17: [1, 4, 0, 4, 4, 1],
            20: [4, 0, 4, 2, 4, 4],
            23: [0, 4, 3, 2, 2, 2],
            26: [2, 3, 2, 1, 2, 3],
            27: [4, 4, 4, 4, 0, 4]
        }
        return emotion_matrix

    @staticmethod
    def emotion_list():
        return ['Angry', 'Fear', 'Sad', 'Happy', 'Surprise', 'Disgust']

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
                for similars in self.similar_arr:
                    for num in similars:
                        if num in au_list_arr[0]:
                            for other_num in [x for x in similars if x is not num]:
                                au_list_arr.append(self.replace(au_list_arr[0], num, other_num))
        for emotion in emotion_templates.keys():
            emotion_templates[emotion] = [sorted(v) for v in emotion_templates[emotion]]

        return emotion_templates

    def replace(self, arr, num, other_num):
        small_arr = [x for x in arr if x is not num]
        small_arr.append(other_num)
        large_set = set(small_arr)
        return list(large_set)

    def emotion_distance(self, nums):
        emote_matrix = self.make_emotion_matrix()
        emote_list = self.emotion_list()
        distance_dict = defaultdict()
        for index, emotion in enumerate(emote_list):
            distance = None
            for num in nums:
                if num in emote_matrix.keys():
                    if not distance:
                        distance = emote_matrix[num][index]
                    else:
                        distance += emote_matrix[num][index]
            distance_dict[emotion] = distance
        return distance_dict

    def make_frame_emotions(self, presence_dict):
        # reverse_frame_dict = defaultdict()
        # for frame, au_dict in presence_dict.items():
        #     reverse_frame_dict[frame] = defaultdict()
        #     for au, amount in au_dict.items():
        #         au_num = int(re.findall("\d+", au)[0])
        #         if 'c' in au:
        #             replace = au.replace('c', 'r')
        #             if replace in au_dict.keys():
        #                 value = au_dict[replace]
        #             else:
        #                 value = 5
        #             if value in reverse_frame_dict[frame].keys():
        #                 reverse_frame_dict[frame][value].append(au_num)
        #             else:
        #                 reverse_frame_dict[frame][value] = [au_num]
        # frame_emotion_dict = defaultdict()
        # for frame, reverse_aus in reverse_frame_dict.items():
        #     aus = [reverse_aus[au_amount] for au_amount in sorted(reverse_aus.keys())]
        #     lcs_dict = self.find_all_lcs(aus)
        #     frame_emotion_dict[frame] = lcs_dict
        # return frame_emotion_dict

        # frame_emotion_dict = {
        #      frame: self.emotion_dict([int(re.findall("\d+", au)[0]) for au in au_dict.keys() if 'c' in au])
        #      for frame, au_dict in presence_dict.items()}

        frame_emotion_dict = {
            frame: self.find_all_lcs(sorted([int(re.findall("\d+", au)[0]) for au in au_dict.keys() if 'c' in au]))
            for frame, au_dict in presence_dict.items()}

        return frame_emotion_dict

    def find_all_lcs(self, aus):
        emote_template = self.emotion_templates()
        return {emotion: max([self.lcs_length(template, aus) for template in template_arr]) for emotion, template_arr in
                emote_template.items()}

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
