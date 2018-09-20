import copy
import glob
import json
import os
import shutil
import subprocess
import sys
from dask import dataframe as df
import cv2
import numpy as np

# import numexpr
# numexpr.set_nthreads(1)

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from OpenFaceScripts.runners import VidCropper, CropAndOpenFace
from OpenFaceScripts.helpers.patient_info import patient_day_session
from OpenFaceScripts.scoring import AUScorer


def make_more_bright(ims, i):
    """
    Makes an image brighter.

    :param ims: List of image names.
    :param i: Index of image within ims.
    """
    name = ims[i]
    im = cv2.imread(name)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype("float64")
    hue, saturation, value = cv2.split(hsv)
    change = 50
    value += np.float64(change)
    value = np.clip(value, 0, 255)
    final_hsv = cv2.merge((hue, saturation, value))
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

        if lines[0] == 'None\n' and lines[1] == 'None\n' and lines[
                2] == 'None\n' and lines[3] == 'None\n':
            return [None, None, None, None, int(lines[5])]

        return {
            'x_min': int(lines[0]),
            'y_min': int(lines[1]),
            'x_max': int(lines[2]),
            'y_max': int(lines[3]),
            'rescale_factor': int(lines[5])
        }


def throw_vid_in_reverse(vid: str, vid_dir: str,
                         include_eyebrows: bool) -> None:
    """
    Reverse a video and run OpenFace on it.
    :param vid_dir: Crop directory for video (created from CropAndOpenFace)
    :return: Dictionary with emotions detected from reversed video
    """
    out_dir = os.path.join(vid_dir, 'reverse')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if 'inter_out.avi' not in os.listdir(out_dir):
        subprocess.Popen(
            "ffmpeg -loglevel quiet -y -i {0} -vf reverse {1}".format(
                vid, os.path.join(out_dir, 'inter_out.avi')),
            shell=True).wait()
    assert 'inter_out.avi' in os.listdir(out_dir)

    # if 'au.csv' not in os.listdir(out_dir):
    CropAndOpenFace.run_open_face(
        out_dir, vid_mode=True, remove_intermediates=False)
    assert os.path.exists(os.path.join(out_dir, 'au.csv'))
    new_df = AUScorer.au_data_frame(out_dir)
    num_new_frames = len(new_df['frame'])

    if new_df:
        new_df = new_df.assign(frame=lambda x: num_new_frames - x['frame'])
    else:
        new_df = None

    return new_df


def re_crop(vid: str, original_crop_coords, scorer: AUScorer.AUScorer,
            out_dir: str) -> dict:
    bounds_dict = presence_bounds(vid, original_crop_coords, scorer)
    min_x = bounds_dict[0]
    min_y = bounds_dict[1]
    max_x = bounds_dict[2]
    max_y = bounds_dict[3]
    width = max_x - min_x
    height = max_y - min_y

    if 'au.csv' not in os.listdir(out_dir):
        VidCropper.crop_and_resize(vid, width, height, min_x, min_y, out_dir,
                                   5)
        CropAndOpenFace.run_open_face(
            out_dir, vid_mode=True, remove_intermediates=False)
    # new_scorer = AUScorer.AUScorer(out_dir)

    # return new_scorer.presence_dict

    return AUScorer.au_data_frame(out_dir)


def presence_bounds(vid: str, original_crop_coords,
                    scorer: AUScorer.AUScorer) -> list:
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
            x_arr = [
                x / rescale_factor for v, x in x_y_dict.items() if 'x_' in v
            ]
            y_arr = [
                y / rescale_factor for v, y in x_y_dict.items() if 'y_' in v
            ]
            min_x = min(x_arr)
            min_y = min(y_arr)
            max_x = max(x_arr)
            max_y = max(y_arr)

            if not bb_arr:
                bb_arr = [min_x, min_y, max_x, max_y]
            else:
                bb_arr = [
                    min(min_x, bb_arr[0]),
                    min(min_y, bb_arr[1]),
                    max(max_x, bb_arr[2]),
                    max(max_y, bb_arr[3])
                ]
    offset = 50

    if not min_x:
        min_x = original_crop_coords['x_min']

    if not max_x:
        max_x = original_crop_coords['x_max']

    if not max_y:
        max_y = original_crop_coords['y_max']

    if not min_y:
        min_y = original_crop_coords['y_min']

    x_arr = np.clip([
        original_crop_coords['x_min'] + min_x - offset,
        original_crop_coords['x_min'] + max_x + offset
    ], 0, vid_width)
    y_arr = np.clip([
        original_crop_coords['y_min'] + min_y - offset,
        original_crop_coords['y_min'] + max_y + offset
    ], 0, vid_height)

    min_x = x_arr[0]
    min_y = y_arr[0]
    max_x = x_arr[1]
    max_y = y_arr[1]

    return [min_x, min_y, max_x, max_y]


def reverse_re_crop_vid_dir(vid: str, vid_dir: str,
                            include_eyebrows: bool) -> dict:
    reverse_vid_dir = os.path.join(vid_dir, 'reverse')
    scorer = AUScorer.AUScorer(reverse_vid_dir, include_eyebrows)

    if scorer.presence_dict:
        out_dir = os.path.join(vid_dir, 'reverse_re_crop')

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        vid = glob.glob(os.path.join(reverse_vid_dir, '*.avi'))[0]
        original_crop_coords = get_dimensions(vid_dir)

        return re_crop(vid, original_crop_coords, scorer, out_dir)


def re_crop_vid_dir(vid, vid_dir, include_eyebrows) -> dict:
    scorer = AUScorer.AUScorer(vid_dir, include_eyebrows)

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
    subprocess.Popen([
        'ffmpeg', '-loglevel', 'quiet', '-y', '-i', vid, '-vf', 'negate',
        os.path.join(out_dir, 'inter_out.avi')
    ]).wait()

    if 'inter_out.avi' in os.listdir(out_dir):
        CropAndOpenFace.run_open_face(out_dir, True, True)

    if 'au.csv' in os.listdir(out_dir):
        # new_scorer = AUScorer.AUScorer(out_dir)

        return AUScorer.au_data_frame(out_dir)

        # return new_scorer.presence_dict

    return None


# def change_gamma(vid: str, vid_dir: str, include_eyebrows: bool) -> dict:
# # TODO: Keep going

# return_dict = {}

# for gamma in [.85, 2]:
# return_dict.update(spec_gamma_change(vid, vid_dir, gamma))

# return return_dict


def lower_gamma(vid: str, vid_dir: str,
                include_eyebrows: bool) -> df.DataFrame:

    return spec_gamma_change(vid, vid_dir, .85)


def increase_gamma(vid: str, vid_dir: str,
                   include_eyebrows: bool) -> df.DataFrame:

    return spec_gamma_change(vid, vid_dir, 2)


def spec_gamma_change(vid: str, vid_dir: str, gamma: float) -> dict:
    out_dir = os.path.join(vid_dir, 'low_gamma')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    subprocess.Popen([
        'ffmpeg', '-loglevel', 'quiet', '-y', '-i', vid, '-vf',
        'eq=gamma={0}'.format(gamma),
        os.path.join(out_dir, 'inter_out.avi')
    ]).wait()

    if 'inter_out.avi' in os.listdir(out_dir):
        CropAndOpenFace.run_open_face(out_dir, True, True)

    if 'au.csv' in os.listdir(out_dir):
        # new_scorer = AUScorer.AUScorer(out_dir)

        return AUScorer.au_data_frame(out_dir)

    return None


def process_vid_dir(eyebrow_dict: dict, vid_dir: str) -> None:
    # all_dict_file = os.path.join(vid_dir, 'all_dict.txt')
    patient_name = vid_dir.split('_')[0]
    all_dict_folder = ('all_' + patient_name)

    already_ran_file = os.path.join(vid_dir, 'already_ran.txt')

    diff_dict = json.load(
        open(already_ran_file)) if os.path.exists(already_ran_file) else {}

    if vid_dir not in diff_dict:
        diff_dict[vid_dir] = {}

    emotion_frame = df.read_hdf(
        os.path.join(all_dict_folder, '*.hdf'), '/data') if os.path.exists(
            all_dict_folder) else AUScorer.au_data_frame(vid_dir)
    # emotion_dict = AUScorer.convert_dict_to_int(
    # json.load(open(all_dict_file))) if os.path.exists(
    # all_dict_file) else AUScorer.AUScorer(vid_dir).presence_dict

    include_eyebrows = eyebrow_dict and vid_dir in eyebrow_dict['Eyebrows']
    pre_func_list = [(re_crop_vid_dir, 're_crop'),
                     (throw_vid_in_reverse, 'reverse'),
                     (reverse_re_crop_vid_dir, 'reverse_re_crop')]

    post_func_list = [(invert_colors, 'invert_colors'),
                      (lower_gamma, 'low_gamma'), (increase_gamma,
                                                   'high_gamma')]

    dir_list = [name for _, name in pre_func_list + post_func_list]

    to_do_list = [x for _, x in pre_func_list if x not in diff_dict[vid_dir]]

    for func, name in pre_func_list + post_func_list:
        if name not in diff_dict[vid_dir]:
            post_func_frame = func(
                get_vid_from_dir(vid_dir), vid_dir, include_eyebrows)
            update_frames(
                post_func_frame=post_func_frame,
                emotion_frame=emotion_frame,
                diff_dict=diff_dict,
                vid_dir=vid_dir,
                name=name,
                func_name='as-is')

    for pre_dir in to_do_list:
        if os.path.exists(os.path.join(vid_dir, pre_dir)):
            if pre_dir not in diff_dict[vid_dir]:
                diff_dict[vid_dir][pre_dir] = {}

            for func, name in post_func_list:
                if name not in diff_dict[vid_dir][pre_dir]:
                    full_path = os.path.join(vid_dir, pre_dir)
                    post_func_frame = func(
                        glob.glob(os.path.join(full_path, '*.avi'))[0],
                        full_path, include_eyebrows)
                    update_frames(post_func_frame, emotion_frame, diff_dict,
                                  vid_dir, pre_dir, name)

    # json.dump(emotion_dict, open(all_dict_file, 'w'))
    json.dump(diff_dict, open(already_ran_file, 'w'))

    for pre_dir in dir_list:
        if os.path.exists(os.path.join(vid_dir, pre_dir)):
            shutil.rmtree(os.path.join(vid_dir, pre_dir))


def get_merged_value(a, b, c, d):
    out_vals = []

    for a, b, c, d in zip(a, b, c, d):
        if pd.isnull(a):
            out_vals.append(b)
        elif pd.isnull(b):
            out_vals.append(a)
        elif c >= d:
            out_vals.append(a)
        elif d >= c:
            out_vals.append(b)

    return pd.Series(out_vals)


def update_frames(post_func_frame: df.DataFrame, emotion_frame: df.DataFrame,
                  diff_dict: dict, vid_dir: str, name: str, func_name: str):

    all_cols = list(emotion_frame.columns)
    merged_cols = ['frame', 'timestamp', 'patient', 'day', 'session']
    other_cols = [x for x in all_cols if x not in merged_cols]
    emotion_frame = emotion_frame.merge(
        post_func_frame, 'outer', merged_cols, suffixes=('_old', '_new'))

    assign_dict = {
        col: get_merged_value(emotion_frame[col + '_old'],
                              emotion_frame[col + '_new'], emotion_frame.B_old,
                              emotion_frame.B_new)
        for col in other_cols
    }
    # print(assign_dict)

    for col in other_cols:
        emotion_frame[col] = assign_dict[col]

    emotion_frame = emotion_frame.drop(
        [x for x in list(emotion_frame.columns) if '_old' in x or '_new' in x],
        axis=1)
    emotion_frame = emotion_frame.compute()

    if name not in diff_dict[vid_dir]:
        diff_dict[vid_dir][name] = {}
        diff_dict[vid_dir][name][func_name] = -1  # TODO: FIX THIS


def get_vid_from_dir(vid_dir: str) -> str:
    """
    Returns the full path to a video associated with a crop directory.

    :param vid_dir: The crop directory.
    :return: The full path to a video associated with a crop directory.
    """

    return os.path.join(
        os.path.dirname(vid_dir), (vid_dir.replace('_cropped', '') + '.avi'))


def process_eyebrows(dir: str, file) -> dict:
    """
    Turn the file containing eyebrow information into eyebrow dict
    :param dir: Folder containing patients
    :param file: Eyebrow file
    :return: eyebrow dict
    """
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
                    exact_dict['Eyebrows'] += [
                        x for x in os.listdir(dir)
                        if os.path.isdir(os.path.join(dir, x)) and line in x
                        and os.path.join(dir,
                                         x) not in exact_dict['No Eyebrows']
                    ]
                else:
                    exact_dict['No Eyebrows'] += [
                        x for x in os.listdir(dir)
                        if os.path.isdir(os.path.join(dir, x)) and line in x
                        and os.path.join(dir,
                                         x) not in exact_dict['No Eyebrows']
                    ]

    for eyebrow_dir in exact_dict['No Eyebrows']:
        if eyebrow_dir in exact_dict['Eyebrows']:
            exact_dict['Eyebrows'].remove(eyebrow_dir)

    return exact_dict


if __name__ == '__main__':
    patient_directory = sys.argv[sys.argv.index('-od') + 1]
    starting_patient_index = sys.argv.index('--')
    patients = sys.argv[starting_patient_index + 1:]
    files = [
        x for x in (os.path.join(patient_directory, vid_dir)
                    for vid_dir in os.listdir(patient_directory))
        if (os.path.isdir(x) and 'au.csv' in os.listdir(x) and any(
            patient in x for patient in patients))
    ]
    # left = sys.argv[sys.argv.index('-vl') + 1]
    # right = sys.argv[sys.argv.index('-vr') + 1]
    eyebrow_file = os.path.join(patient_directory, 'eyebrows.txt')
    eyebrow_dict = process_eyebrows(
        patient_directory,
        open(eyebrow_file)) if os.path.exists(eyebrow_file) else {}

    if eyebrow_dict:
        json.dump(
            eyebrow_dict,
            open(os.path.join(patient_directory, 'eyebrow_dict.txt'), 'w'))

    # TODO: better multiprocessing on this

    for vid_dir in files:
        process_vid_dir(eyebrow_dict, vid_dir)
