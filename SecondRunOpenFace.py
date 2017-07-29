"""
.. module SecondRunOpenFace
    :synopsis: Module for use after an initial run of OpenFace on a video set, attempts to rerun on the videos
        that OpenFace could not recognize a face in the first time.
"""
import glob
import json
import os
import shutil
import sys
import subprocess
from collections import defaultdict
import copy

from pathos.multiprocessing import ProcessingPool as Pool

import cv2
import numpy as np

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import AUScorer, CropAndOpenFace, VidCropper


def make_more_bright(ims, i):
    """
    Makes an image brighter.

    :param ims: List of image names.
    :param name: Name of image
    """
    name = ims[i]
    im = cv2.imread(name)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype("float64")
    h, s, v = cv2.split(hsv)
    change = 50
    v += np.float64(change)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    im = cv2.cvtColor(final_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    cv2.imwrite(name, im)
    print(name)


def height_width(vid_file_path):
    """
    Gets height and width of a video.
    :param vid_file_path: Path to video.
    :return: [Vid height, Vid width]
    """
    _json = VidCropper.probe(vid_file_path)

    if 'streams' in _json:
        height = None
        width = None
        for s in _json['streams']:
            if 'height' in s:
                height = s['height']
            if 'width' in s:
                width = s['width']
            if height and width:
                return [height, width]

    raise Exception('No Height and Width found')


def get_dimensions(vid_dir):
    with open(os.path.join(vid_dir, 'bb_arr.txt')) as bb_file:
        lines = bb_file.readlines()
        if lines[0] == 'None\n' and lines[1] == 'None\n' and lines[2] == 'None\n' and lines[3] == 'None\n':
            return [None, None, None, None, int(lines[5])]
        return {
            'x_min': int(lines[0]),
            'y_min': int(lines[1]),
            'x_max': int(lines[2]),
            'y_max': int(lines[3]),
            'rescale_factor': int(lines[5])
        }


class OpenFaceSecondRunner:
    """
    Main runner class
    """

    def __init__(self, patient_directory):
        """
        Default constructor.

        :param patient_directory: Directory in which OpenFace was run.
        """
        self.re_crop_file = os.path.join(patient_directory, 're_crop.txt')
        self.reverse_re_crop_file = os.path.join(patient_directory, 'reverse_re_crop.txt')
        self.reverse_file = os.path.join(patient_directory, 'fr_files.txt')
        self.crop_diff = defaultdict()
        self.reverse_crop_diff = defaultdict()
        self.fr_dict = defaultdict()

        if os.path.exists(self.re_crop_file):
            self.crop_diff = json.load(open(self.re_crop_file))
        if os.path.exists(self.reverse_re_crop_file):
            self.reverse_crop_diff = json.load(open(self.reverse_re_crop_file))
        if os.path.exists(self.reverse_file):
            self.fr_dict = json.load(open(self.reverse_file))

        files = [x for x in (os.path.join(patient_directory, vid_dir) for vid_dir in os.listdir(patient_directory) if
                             os.path.isdir(os.path.join(patient_directory, vid_dir))) if 'au.txt' in os.listdir(x)]
        Pool().map(self.process_vid_dir, files)

        json.dump(self.crop_diff, open(self.re_crop_file, 'w'))
        json.dump(self.fr_dict, open(self.reverse_file, 'w'))
        json.dump(self.reverse_crop_diff, open(self.reverse_re_crop_file), 'w')

    def process_vid_dir(self, vid_dir):
        print(vid_dir)
        orig_dict = AUScorer.AUScorer(vid_dir).emotions
        if vid_dir not in self.crop_diff:
            re_crop_dict = self.re_crop_vid_dir(vid_dir, self.crop_diff)
            if re_crop_dict:
                orig_dict.update(re_crop_dict)
        if vid_dir not in self.fr_dict:
            reverse_dict = self.throw_vid_in_reverse(vid_dir)
            if reverse_dict:
                orig_dict.update(reverse_dict)
        if os.path.join(vid_dir, 'reverse') not in self.reverse_crop_diff:
            reverse_re_crop_dict = self.reverse_re_crop_vid_dir(vid_dir, self.reverse_crop_diff)
            if reverse_re_crop_dict:
                orig_dict.update(reverse_re_crop_dict)
        json.dump(orig_dict, open(os.path.join(vid_dir, 'all_dict.txt'), 'w'))

    def re_crop_vid_dir(self, vid_dir, crop_diff):
        scorer = AUScorer.AUScorer(vid_dir, 0, False)
        if scorer.emotions:
            out_dir = os.path.join(vid_dir, 're_crop')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            vid = get_vid_from_dir(vid_dir)
            original_crop_coords = get_dimensions(vid_dir)
            return self.re_crop(vid, vid_dir, original_crop_coords, scorer, out_dir, crop_diff)

    def reverse_re_crop_vid_dir(self, vid_dir, crop_diff):
        reverse_vid_dir = os.path.join(vid_dir, 'reverse')
        scorer = AUScorer.AUScorer(reverse_vid_dir, 0, False)
        if scorer.emotions:
            out_dir = os.path.join(reverse_vid_dir, 're_crop')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            vid = glob.glob(os.path.join(reverse_vid_dir, '*.avi'))[0]
            original_crop_coords = get_dimensions(vid_dir)
            return self.re_crop(vid, vid_dir, original_crop_coords, scorer, out_dir, crop_diff)

    @staticmethod
    def re_crop(vid, vid_dir, original_crop_coords, scorer, out_dir, crop_diff):
        vid_height, vid_width = height_width(vid)
        min_x = None
        max_x = None
        min_y = None
        max_y = None

        if 'x_min' not in original_crop_coords:
            original_crop_coords = {
                'x_min': 0,
                'y_min': 0,
                'x_max': vid_width,
                'y_max': vid_height,
                'rescale_factor': original_crop_coords[4]
            }
        bb_arr = []  # min_x, min_y, max_x, max_y
        for frame in scorer.emotions:
            if frame in scorer.x_y_dict:
                rescale_factor = original_crop_coords['rescale_factor']
                x_y_dict = scorer.x_y_dict[frame]
                x_arr = [x / rescale_factor for v, x in x_y_dict.items() if 'x_' in v]
                y_arr = [y / rescale_factor for v, y in x_y_dict.items() if 'y_' in v]
                min_x = min(x_arr)
                min_y = min(y_arr)
                max_x = max(x_arr)
                max_y = max(y_arr)
                if not bb_arr:
                    bb_arr = [min_x, min_y, max_x, max_y]
                else:
                    bb_arr = [min(min_x, bb_arr[0]), min(min_y, bb_arr[1]), max(max_x, bb_arr[2]),
                              max(max_y, bb_arr[3])]
        offset = 50

        if not min_x:
            min_x = original_crop_coords['x_min']
        if not max_x:
            max_x = original_crop_coords['x_max']
        if not max_y:
            max_y = original_crop_coords['y_max']
        if not min_y:
            min_y = original_crop_coords['y_min']

        x_arr = np.clip(
            [original_crop_coords['x_min'] + min_x - offset, original_crop_coords['x_min'] + max_x + offset], 0,
            vid_width)
        y_arr = np.clip(
            [original_crop_coords['y_min'] + min_y - offset, original_crop_coords['y_min'] + max_y + offset], 0,
            vid_height)

        min_x = x_arr[0]
        min_y = y_arr[0]
        max_x = x_arr[1]
        max_y = y_arr[1]
        width = max_x - min_x
        height = max_y - min_y
        if 'au.txt' not in os.listdir(out_dir):
            VidCropper.crop_and_resize(vid, width, height, min_x, min_y, out_dir, 5)
            CropAndOpenFace.run_open_face(out_dir, vid_mode=True, remove_intermediates=True)
        new_scorer = AUScorer.AUScorer(out_dir)
        x = len(new_scorer.emotions) - len(scorer.emotions)
        crop_diff[vid_dir] = x
        return new_scorer.emotions

    def throw_vid_in_reverse(self, vid_dir):
        vid = get_vid_from_dir(vid_dir)
        out_dir = os.path.join(vid_dir, 'reverse')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if 'au.txt' not in os.listdir(out_dir):
            subprocess.Popen("ffmpeg -y -i {0} -vf reverse {1}".format(vid, os.path.join(out_dir, 'inter_out.avi')),
                             shell=True).wait()
            CropAndOpenFace.run_open_face(out_dir, vid_mode=True, remove_intermediates=True)
        old_scorer = AUScorer.AUScorer(vid_dir)
        new_scorer = AUScorer.AUScorer(out_dir)
        num_frames = int(VidCropper.duration(os.path.join(vid_dir, 'out.avi')) * 30)
        new_dict = None
        if len(old_scorer.emotions) > 0 or len(new_scorer.emotions) > 0:
            new_dict = {num_frames - i: k for i, k in new_scorer.emotions.items()}
            old_dict = copy.copy(old_scorer.emotions)
            old_dict.update(new_dict)
            json.dump(new_dict, open(os.path.join(vid_dir, 'fr_dict.txt'), 'w'))
        self.fr_dict[vid_dir] = True
        return new_dict


def get_vid_from_dir(vid_dir):
    """
    Returns the full path to a video associated with a crop directory.

    :param vid_dir: The crop directory.
    :return: The full path to a video associated with a crop directory.
    """
    return os.path.join(directory, (vid_dir.replace('_cropped', '') + '.avi'))


if __name__ == '__main__':
    directory = sys.argv[sys.argv.index('-od') + 1]
    OpenFaceSecondRunner(directory)
