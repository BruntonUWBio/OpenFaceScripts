import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np


class OpenFaceScorer:
    def __init__(self, dir, csv):
        os.chdir(dir)
        self.include_guess = True  # Change depending on desired behavior
        self.landmark_map = self.make_landmark_map()
        au_file = 'au.txt'  # Replace with name of action units file
        self.au_arr, self.au_dict = self.make_au_parts(au_file)
        self.ref_dict = self.csv_reader(csv)  # Open coordinates file for reference
        self.original_file_names = sorted(self.ref_dict.keys())
        self.im_files = self.find_im_files(dir)
        self.coords_dict = {}
        with open('bb_arr.txt') as f:
            f_arr = f.readlines()
            self.bbox = f_arr[0:3]
            self.bbox = [int(i) for i in self.bbox]
            self.rescale_factor = int(f_arr[5])
        self.normalize_au_coords()
        all_frame_nums = [i for i in self.coords_dict.keys()]
        self.part_arr = [
            "Left Eye",
            "Right Eye",
            "Left Eyebrow",
            "Right Eyebrow",
            "Nose",
            "Mouth",
            "Jaw"
        ]
        cut_frames = [self.coords_dict[i] for i in range(min(all_frame_nums), max(all_frame_nums), 30)]
        scores_list = defaultdict()
        for part in self.part_arr:
            scores_list[part] = []
        detectedNum = 0
        totalNum = 0
        for index, i in enumerate(cut_frames):
            totalNum += 1
            if np.count_nonzero(i):
                detectedNum += 1
                arr1 = i
                arr2 = [self.ref_dict[self.original_file_names[index]][j] for j in
                        self.ref_dict[self.original_file_names[index]].keys()]
                curr_scores_dict = self.find_score_diffs(arr1, arr2)
                for part, score in curr_scores_dict.items():
                    scores_list[part].append(score)
        detectedPercentage = detectedNum/totalNum
        scores_list = self.reduce_to_averages(scores_list)
        with open('av_score.txt', mode='w') as f:
            print('Detected percentage: ' + str(detectedPercentage))
            f.write(str(detectedPercentage))
            for part in sorted(scores_list.keys()):
                score_string = part + " = " + str(scores_list[part])
                print(score_string)
                f.write(score_string)
                f.write('\n')

    # Returns png files within the path specified, non-recursively
    @staticmethod
    def find_im_files(path):
        return sorted(
            glob.glob(os.path.join(path + '/*.png')))  # Sort, because order of images matters

    @staticmethod
    def reduce_to_averages(dict):
        for part in dict.keys():
            dict[part] = np.average(dict[part])
        return dict

    def find_score_diffs(self, arr1, arr2):
        dist_arr = defaultdict()
        for part in self.part_arr:
            dist_arr[part] = []
        for index, coord_arr1 in enumerate(arr1):
            if index in range(len(arr2)):
                coord_arr2 = arr2[index]
                x1 = coord_arr1[0]
                x2 = coord_arr2[0]
                y1 = coord_arr1[1]
                y2 = coord_arr2[1]
                if x1 and x2 and y1 and y2:  # Check if any nones
                    if index in range(1, 18):
                        part = "Jaw"
                    elif index in range(18, 23):
                        part = "Right Eyebrow"
                    elif index in range(23, 28):
                        part = "Left Eyebrow"
                    elif index in range(28, 37):
                        part = "Nose"
                    elif index in range(37, 43):
                        part = "Right Eye"
                    elif index in range(43, 49):
                        part = "Left Eye"
                    else:
                        part = "Mouth"
                    dist_arr[part].append(self.find_distance(x1, x2, y1, y2))
        return self.reduce_to_averages(dist_arr)

    @staticmethod
    def find_distance(x1, x2, y1, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def normalize_au_coords(self):
        xmin = self.bbox[0]
        ymin = self.bbox[1]
        min_arr = [xmin, ymin]
        for frame_index in range(len(self.au_arr)):
            self.coords_dict[frame_index] = []
            for face_coord in range(0, 68):
                if self.au_arr[frame_index][self.au_dict['success']] == '0':
                    x_coord = 0
                    y_coord = 0
                else:
                    x_coord = float(self.au_arr[frame_index][self.au_dict['x_' + str(face_coord)]])
                    y_coord = float(self.au_arr[frame_index][self.au_dict['y_' + str(face_coord)]])

                    # Adjust coordinates to where they would have been on the original video
                    x_coord /= self.rescale_factor
                    y_coord /= self.rescale_factor
                    x_coord += xmin
                    y_coord += ymin
                self.coords_dict[frame_index].append([x_coord, y_coord])

    # From XMLTransformer
    def make_landmark_map(self):
        landmark_map = defaultdict()
        for i in range(0, 17):
            landmark_map[i] = 'J' + str(i + 1)
        landmark_map[17] = 'E10'
        landmark_map[18] = 'E9'
        landmark_map[19] = 'E8'
        landmark_map[20] = 'E7'
        landmark_map[21] = 'E6'
        for i in range(22, 27):
            landmark_map[i] = 'E' + str(i - 21)
        for i in range(27, 36):
            landmark_map[i] = 'N' + str(i - 26)
        for i in range(36, 39):
            landmark_map[i] = 'RE' + str(i - 35)
        landmark_map[39] = 'RE6'
        landmark_map[40] = 'RE5'
        landmark_map[41] = 'RE4'
        for i in range(42, 45):
            landmark_map[i] = 'LE' + str(i - 41)
        landmark_map[45] = 'LE6'
        landmark_map[46] = 'LE5'
        landmark_map[47] = 'LE4'
        for i in range(48, 55):
            landmark_map[i] = 'M' + str(i - 47)
        landmark_map[55] = 'M6'
        landmark_map[56] = 'M7'
        landmark_map[57] = 'M8'
        landmark_map[58] = 'M9'
        landmark_map[59] = 'M10'
        landmark_map[60] = 'M11'
        landmark_map[61] = 'M12'
        landmark_map[62] = 'M13'
        landmark_map[63] = 'M14'
        landmark_map[64] = 'M15'
        landmark_map[65] = 'M19'
        landmark_map[66] = 'M18'
        landmark_map[67] = 'M17'
        return landmark_map

    @staticmethod
    def make_au_parts(au_file):
        with open(au_file, mode='r') as f:
            au_arr = f.readlines()
        au_arr = [au_arr[i].replace('\n', '').split(', ') for i in range(0, len(au_arr))]
        au_dict = {label: ind for ind, label in enumerate(au_arr[0])}
        au_arr = au_arr[1: len(au_arr)]
        for frame in range(len(au_arr)):
            for i in range(len(au_arr[frame])):
                au_arr[frame][i] = float(au_arr[frame][i])
        return au_arr, au_dict

    # csv_reader, from XMLTransformer
    def csv_reader(self, csv_path):
        image_map = defaultdict()
        first_row = None
        split_path = os.path.dirname(csv_path)
        with open(csv_path, 'rt') as csv_file:
            reader = csv.reader(csv_file)
            for index, row in enumerate(reader):
                if index == 0:
                    first_row = row
                else:
                    filename = os.path.join(split_path, row[0])
                    image_map[filename] = defaultdict()
                    for j in range(68):
                        part_num = self.landmark_map[j]
                        ind = first_row.index(part_num)
                        if ind < len(row):
                            try:
                                if row[ind + 2] == '':
                                    pass
                                else:
                                    if int(float(row[ind + 2])) == 0 or self.include_guess is True:
                                        x = abs(float(row[ind]))
                                        y = abs(float(row[ind + 1]))
                                        image_map[filename][j] = []
                                        if x == 1.0 and y == 1.0:
                                            image_map[filename][j].append(None)
                                            image_map[filename][j].append(None)
                                        else:
                                            image_map[filename][j].append(x)
                                            image_map[filename][j].append(y)
                            except ValueError as e:
                                print(e)
                                print(csv_path + ' Has faulty encoding')
        return image_map
