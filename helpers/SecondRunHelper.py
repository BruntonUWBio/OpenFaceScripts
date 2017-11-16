import functools
import json
import sys

import copy
import os
import subprocess
import shutil
import numpy as np
import cv2

from OpenFaceScripts.runners import VidCropper, CropAndOpenFace
from scoring.AUScorer import AUScorer


def make_more_bright(ims, i):
    """
    Makes an image brighter.

    :param ims: List of image names.
    :param i: Index of image within ims.
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


def get_dimensions(vid_dir: str):
    """
    Gets the crop dimensions from the text file within a directory

    :param vid_dir: Directory
    :return: Either [None, None, None, None, rescale_factor] if no crop or dictionary containing min, max, and rescale
    """
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


def throw_vid_in_reverse(vid: str, vid_dir: str, include_eyebrows: bool) -> None:
    """
    Reverse a video and run OpenFace on it.
    :param vid_dir: Crop directory for video (created from CropAndOpenFace)
    :return: Dictionary with emotions detected from reversed video
    """
    out_dir = os.path.join(vid_dir, 'reverse')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if 'inter_out.avi' not in os.listdir(out_dir):
        subprocess.Popen("ffmpeg -y -i {0} -vf reverse {1}".format(vid, os.path.join(out_dir, 'inter_out.avi')),
                         shell=True).wait()
    assert 'inter_out.avi' in os.listdir(out_dir)
    if 'au.txt' not in os.listdir(out_dir):
        CropAndOpenFace.run_open_face(out_dir, vid_mode=True, remove_intermediates=False)
    assert os.path.exists(os.path.join(out_dir, 'au.txt'))
    old_scorer = AUScorer.AUScorer(vid_dir, 0, include_eyebrows)
    new_scorer = AUScorer.AUScorer(out_dir, 0, include_eyebrows)
    num_frames = int(VidCropper.duration(os.path.join(out_dir, 'inter_out.avi')) * 30)
    new_dict = None
    if len(old_scorer.presence_dict) > 0 or len(new_scorer.presence_dict) > 0:
        new_dict = {num_frames - i: k for i, k in new_scorer.presence_dict.items()}
        old_dict = copy.copy(old_scorer.presence_dict)
        new_dict.update(old_dict)
    return new_dict


def re_crop(vid: str, original_crop_coords, scorer: AUScorer.AUScorer, out_dir: str) -> dict:
    bounds_dict = presence_bounds(vid, original_crop_coords, scorer)
    min_x = bounds_dict[0]
    min_y = bounds_dict[1]
    max_x = bounds_dict[2]
    max_y = bounds_dict[3]
    width = max_x - min_x
    height = max_y - min_y
    if 'au.txt' not in os.listdir(out_dir):
        VidCropper.crop_and_resize(vid, width, height, min_x, min_y, out_dir, 5)
        CropAndOpenFace.run_open_face(out_dir, vid_mode=True, remove_intermediates=False)
    new_scorer = AUScorer.AUScorer(out_dir)
    return new_scorer.presence_dict


def presence_bounds(vid: str, original_crop_coords, scorer: AUScorer.AUScorer) -> list:
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
    for frame in scorer.presence_dict:
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
    return [min_x, min_y, max_x, max_y]


def reverse_re_crop_vid_dir(vid: str, vid_dir: str, include_eyebrows: bool) -> dict:
    reverse_vid_dir = os.path.join(vid_dir, 'reverse')
    scorer = AUScorer.AUScorer(reverse_vid_dir, 0, include_eyebrows)
    if scorer.presence_dict:
        out_dir = os.path.join(vid_dir, 'reverse_re_crop')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        vid = glob.glob(os.path.join(reverse_vid_dir, '*.avi'))[0]
        original_crop_coords = get_dimensions(vid_dir)
        return re_crop(vid, original_crop_coords, scorer, out_dir)


def re_crop_vid_dir(vid, vid_dir, include_eyebrows) -> dict:
    scorer = AUScorer.AUScorer(vid_dir, 0, include_eyebrows)
    if scorer.presence_dict:
        out_dir = os.path.join(vid_dir, 're_crop')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        original_crop_coords = get_dimensions(vid_dir)
        return re_crop(vid, original_crop_coords, scorer, out_dir)


def invert_colors(vid: str, vid_dir: str, include_eyebrows: bool) -> dict:
    out_dir = os.path.join(vid_dir, 'invert_colors')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return invert(vid, out_dir)


def invert(vid: str, out_dir: str) -> dict:
    subprocess.Popen(
        ['ffmpeg', '-y', '-i', vid, '-vf', 'negate', os.path.join(out_dir, 'inter_out.avi')]).wait()
    if 'inter_out.avi' in os.listdir(out_dir):
        CropAndOpenFace.run_open_face(out_dir, True, True)
    if 'au.txt' in os.listdir(out_dir):
        new_scorer = AUScorer.AUScorer(out_dir)
        return new_scorer.presence_dict


def change_gamma(vid: str, vid_dir: str, include_eyebrows: bool) -> dict:
    return_dict = {}
    for gamma in [.85, 2]:
        return_dict.update(spec_gamma_change(vid, vid_dir, gamma))
    return return_dict


def spec_gamma_change(vid: str, vid_dir: str, gamma: float) -> dict:
    out_dir = os.path.join(vid_dir, 'low_gamma')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    subprocess.Popen(
        ['ffmpeg', '-y', '-i', vid, '-vf', 'eq=gamma={0}'.format(gamma), os.path.join(out_dir, 'inter_out.avi')]).wait()
    if 'inter_out.avi' in os.listdir(out_dir):
        CropAndOpenFace.run_open_face(out_dir, True, True)
    if 'au.txt' in os.listdir(out_dir):
        new_scorer = AUScorer.AUScorer(out_dir)
        return new_scorer.presence_dict
    return {}


def process_vid_dir(eyebrow_dict: dict, vid_dir: str) -> None:
    all_dict_file = os.path.join(vid_dir, 'all_dict.txt')
    already_ran_file = os.path.join(vid_dir, 'already_ran.txt')

    diff_dict = json.load(open(already_ran_file)) if os.path.exists(already_ran_file) else {}
    if vid_dir not in diff_dict:
        diff_dict[vid_dir] = {}
    emotion_dict = json.load(open(all_dict_file)) if os.path.exists(all_dict_file) else AUScorer.AUScorer(
        vid_dir).presence_dict
    if vid_dir in eyebrow_dict['Eyebrows']:
        include_eyebrows = True
    else:
        include_eyebrows = False
    pre_func_list = [
        (re_crop_vid_dir, 're_crop'),
        (throw_vid_in_reverse, 'reverse'),
        (reverse_re_crop_vid_dir, 'reverse_re_crop')]

    post_func_list = [
        (invert_colors, 'invert_colors'),
        (change_gamma, 'low_gamma')
    ]

    dir_list = [name for _, name in pre_func_list + post_func_list]

    to_do_list = [x for _, x in pre_func_list if x not in diff_dict[vid_dir]]

    for func, name in pre_func_list + post_func_list:
        if name not in diff_dict[vid_dir]:
            post_func_dict = func(get_vid_from_dir(vid_dir), vid_dir, include_eyebrows)
            update_dicts(post_func_dict=post_func_dict, emotion_dict=emotion_dict, diff_dict=diff_dict, vid_dir=vid_dir,
                         name=name, func_name='as-is')

    for pre_dir in to_do_list:
        if os.path.exists(os.path.join(vid_dir, pre_dir)):
            if pre_dir not in diff_dict[vid_dir]:
                diff_dict[vid_dir][pre_dir] = {}
            for func, name in post_func_list:
                if name not in diff_dict[vid_dir][pre_dir]:
                    full_path = os.path.join(vid_dir, pre_dir)
                    post_func_dict = func(glob.glob(os.path.join(full_path, '*.avi'))[0], full_path, include_eyebrows)
                    update_dicts(post_func_dict, emotion_dict, diff_dict, vid_dir, pre_dir, name)

    json.dump(emotion_dict, open(all_dict_file, 'w'))
    json.dump(diff_dict, open(already_ran_file, 'w'))
    for pre_dir in dir_list:
        if os.path.exists(os.path.join(vid_dir, pre_dir)):
            shutil.rmtree(os.path.join(vid_dir, pre_dir))


def update_dicts(post_func_dict: dict, emotion_dict: dict, diff_dict: dict, vid_dir: str, name: str, func_name: str):
    diff = len([x for x in post_func_dict if x not in emotion_dict]) if post_func_dict else 0
    if post_func_dict:
        for frame in post_func_dict:
            if frame not in emotion_dict or not emotion_dict[frame]:
                emotion_dict[frame] = post_func_dict[frame]
    if name not in diff_dict[vid_dir]:
        diff_dict[vid_dir][name] = {}
    diff_dict[vid_dir][name][func_name] = diff


def get_vid_from_dir(vid_dir: str) -> str:
    """
    Returns the full path to a video associated with a crop directory.

    :param vid_dir: The crop directory.
    :return: The full path to a video associated with a crop directory.
    """
    return os.path.join(os.path.dirname(vid_dir), (vid_dir.replace('_cropped', '') + '.avi'))


def process_eyebrows(dir: str, file) -> dict:
    exact_dict = {'Eyebrows': [], 'No Eyebrows': []}
    lines = file.read().splitlines()
    if lines[0] == "eyebrows:":
        eyebrow_mode = True
        for line in (x for x in lines if x):
            if line == "no eyebrows:":
                eyebrow_mode = False
            crop_dir = os.path.join(dir, line + '_cropped')
            if os.path.exists(crop_dir):
                if eyebrow_mode:
                    exact_dict['Eyebrows'].append(crop_dir)
                else:
                    exact_dict['No Eyebrows'].append(crop_dir)
        for line in (x for x in lines if x):
            if line == "no eyebrows:":
                eyebrow_mode = False
            crop_dir = os.path.join(dir, line + '_cropped')
            if not os.path.exists(crop_dir):
                if eyebrow_mode:
                    exact_dict['Eyebrows'] += [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))
                                               and line in x and os.path.join(dir, x) not in exact_dict['No Eyebrows']]
                else:
                    exact_dict['No Eyebrows'] += [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))
                                                  and line in x and os.path.join(dir, x) not in exact_dict[
                                                      'No Eyebrows']]
    for eyebrow_dir in exact_dict['No Eyebrows']:
        if eyebrow_dir in exact_dict['Eyebrows']:
            exact_dict['Eyebrows'].remove(eyebrow_dir)
    return exact_dict


if __name__ == '__main__':
    patient_directory = sys.argv[sys.argv.index('-od') + 1]

    files = [x for x in (os.path.join(patient_directory, vid_dir) for vid_dir in os.listdir(patient_directory)) if
             (os.path.isdir(x) and 'au.txt' in os.listdir(x))]

    eyebrow_file = os.path.join(patient_directory, 'eyebrows.txt')
    eyebrow_dict = process_eyebrows(patient_directory, open(eyebrow_file)) if os.path.exists(eyebrow_file) else {}
    f = functools.partial(process_vid_dir, eyebrow_dict)
