"""
.. module SecondRunOpenFace
    :synopsis: Module for use after an initial run of OpenFace on a video set, attempts to rerun on the videos
        that OpenFace could not recognize a face in the first time.
"""
import json
import os
import sys
import subprocess
from collections import defaultdict

import cv2
import functools
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import AUScorer, CropAndOpenFace, VidCropper


def make_more_bright(ims, i):
    """
    Makes an image brighter.

    :param ims: List of image names
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
    Gets height and width of a video
    :param vid_file_path: Path to video
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

    def __init__(self, directory):
        """
        Default constructor.

        :param directory: Directory in which OpenFace was run.
        """
        vid_dirs = (os.path.join(directory, vid_dir) for vid_dir in os.listdir(directory) if
                    os.path.isdir(os.path.join(directory, vid_dir)))
        re_crop_file = os.path.join(directory, 're_crop.txt')
        crop_diff = defaultdict()
        if os.path.exists(re_crop_file):
            crop_diff = json.load(open(re_crop_file))

        for vid_dir in (x for x in vid_dirs if x not in crop_diff):
            if 'au.txt' in os.listdir(vid_dir):
                scorer = AUScorer.AUScorer(vid_dir, 0, False)
                if scorer.emotions:
                    out_dir = os.path.join(vid_dir, 're_crop')
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)
                    vid = os.path.join(directory, (vid_dir.replace('_cropped', '') + '.avi'))
                    vid_height, vid_width = height_width(vid)
                    original_crop_coords = get_dimensions(vid_dir)
                    if 'x_min' not in original_crop_coords:
                        original_crop_coords = {
                            'x_min' : 0,
                            'y_min' : 0,
                            'x_max' : vid_width,
                            'y_max' : vid_height,
                            'rescale_factor' : original_crop_coords[4]
                        }
                    bb_arr = []  # min_x, min_y, max_x, max_y
                    for frame in scorer.emotions:
                        if frame in scorer.x_y_dict:
                            rescale_factor = original_crop_coords['rescale_factor']
                            x_y_dict = scorer.x_y_dict[frame]
                            x_arr = [x/rescale_factor for v, x in x_y_dict.items() if 'x_' in v]
                            y_arr = [y/rescale_factor for v, y in x_y_dict.items() if 'y_' in v]
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
                    json.dump(crop_diff, open(re_crop_file, 'w'))
                    # if not scorer.emotions:
                    #     vid = os.path.join(vid_dir, 'out.avi')
                    #     hsv_changed_dir = os.path.join(os.path.dirname(vid), 'hsv_changed')
                    #     if not os.path.exists(hsv_changed_dir):
                    #         os.mkdir(hsv_changed_dir)
                    #     subprocess.Popen(
                    #         'ffmpeg -y -i "{0}" -q:v 2 -vf fps=30 "{1}"'.format(vid, os.path.join(hsv_changed_dir, (
                    #             os.path.basename(vid) + '_out%04d.png'))), shell=True).wait()
                    #     p = Pool()
                    #     pngs = [os.path.join(hsv_changed_dir, x) for x in os.listdir(hsv_changed_dir) if '.png' in x]
                    #     f = functools.partial(make_more_bright, pngs)
                    #     p.map(f, range(len(pngs)))
                    #     CropAndOpenFace.run_open_face(hsv_changed_dir)
                    #     new_scorer = AUScorer.AUScorer(hsv_changed_dir)
                    #     if not new_scorer.emotions:
                    #         log.write(vid_dir + 'has been recognized! \n')
                    #         log.flush()
                    #     else:
                    #         log.write('No change for ' + vid_dir + '\n')
                    #         log.flush()


if __name__ == '__main__':
    directory = sys.argv[sys.argv.index('-od') + 1]
    OpenFaceSecondRunner(directory)
