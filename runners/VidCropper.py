"""
.. module:: VidCropper
    :synopsis: A class for cropping a video given bounding boxes. Aims to be much more performant than ImageCropper for essentially the same purpose.
"""
import json
import os
import subprocess

import numpy as np
from OpenFaceScripts.runners import SecondRunOpenFace, CropAndOpenFace
from OpenFaceScripts.scoring.AUScorer import AUScorer


def crop_and_resize(vid, width, height, x_min, y_min, directory, resize_factor):
    """
    Crops a video and then resizes it

    :param vid: Video to crop
    :param width: Width of crop
    :type width: Union[int, float]
    :param height: Height of crop
    :type height: Union[int, float]
    :param x_min: x-coordinate of top-left corner
    :type x_min: Union[int, float]
    :param y_min: y-coordinate of top-left corner
    :type y_min: Union[int, float]
    :param directory: Directory to output files to
    :param resize_factor: Factor by which to resize the cropped video
    """
    crop_vid = os.path.join(directory, 'cropped_out.avi')
    subprocess.Popen(
        'ffmpeg -y -loglevel quiet -i {0} -filter:v \"crop={1}:{2}:{3}:{4}\" {5}'.format(vid, str(width), str(height),
                                                                                         str(x_min), str(y_min),
                                                                                         crop_vid),
        shell=True).wait()
    subprocess.Popen(
        'ffmpeg -y -loglevel quiet -i {0} -vf scale={2}*iw:{2}*ih {1}'.format(crop_vid,
                                                                              os.path.join(directory, 'inter_out.avi'),
                                                                              str(resize_factor)), shell=True).wait()
    os.remove(os.path.join(directory, 'cropped_out.avi'))


class DurationException(Exception):
    def __init__(self, string):
        Exception.__init__(self, string)


def duration(vid_file_path):
    ''' Video's duration in seconds, return a float number
    '''
    _json = probe(vid_file_path)

    if 'format' in _json:
        if 'duration' in _json['format']:
            return float(_json['format']['duration'])

    if 'streams' in _json:
        # commonly stream 0 is the video
        for s in _json['streams']:
            if 'duration' in s:
                return float(s['duration'])

    # if everything didn't happen,
    # we got here because no single 'return' in the above happen.
    raise DurationException('I found no duration')
    # return None


def find_crop_path(file, crop_txt_files):
    parts = file.split('.')
    pid = parts[0]
    out_file = None
    if pid in crop_txt_files:
        out_file = crop_txt_files[pid]
    return out_file


class CropVid:
    """
    Main cropper class
    """

    def __init__(self, vid, directory, crop_txt_files, nose_txt_files):
        self.resize_factor = 5
        self.fps_fraction = 1
        self.crop_txt_files = crop_txt_files
        self.nose_txt_files = nose_txt_files
        self.crop_file_path = None
        self.nose_file_path = None
        self.im_dir = directory
        self.vid = vid
        self.vid_width = 640
        self.vid_height = 480
        self.fps = 30
        self.vid_length = duration(vid)
        self.num_frames = int(self.vid_length * self.fps)
        self.read_arr_dict = {}
        self.un_cropped_ims = []
        self.crop_im_arr_arr_dict = self.crop_im_arr_arr_list()
        bb_arr = [None, None, None, None]
        if not all(self.crop_im_arr_arr_dict):
            bb_arr = [50, 0, 640 - 100, 480 - 100]
        else:
            for coords in self.crop_im_arr_arr_dict:
                im_crop_arr = coords
                x_min = im_crop_arr[0]
                y_min = im_crop_arr[1]
                x_max = im_crop_arr[2]
                y_max = im_crop_arr[3]
                if bb_arr[0]:
                    bb_arr = [min(bb_arr[0], x_min), min(bb_arr[1], y_min),
                              max(bb_arr[2], x_max), max(bb_arr[3], y_max)]
                else:
                    bb_arr = [x_min, y_min, x_max, y_max]
        if not all(bb_arr):
            bb_arr = [50, 0, 640 - 100, 480 - 100]
        x_min = bb_arr[0]
        y_min = bb_arr[1]
        x_max = bb_arr[2]
        y_max = bb_arr[3]

        self.write_arr(bb_arr, 'bb_arr', extra=True)
        width = x_max - x_min
        height = y_max - y_min

        crop_and_resize(vid, width, height, x_min, y_min, directory, resize_factor=self.resize_factor)

    def crop_im_arr_arr_list(self):
        base_name = os.path.basename(self.vid)
        self.crop_file_path = find_crop_path(base_name, self.crop_txt_files)
        self.nose_file_path = find_crop_path(base_name, self.nose_txt_files)
        crop_read_arr = self.make_read_arr(open(self.crop_file_path)) if self.crop_file_path else None
        nose_read_arr = self.make_read_arr(open(self.nose_file_path)) if self.nose_file_path else None
        if crop_read_arr and nose_read_arr:
            original_crop_coords = self.find_im_bb(crop_read_arr, nose_read_arr)
            if not original_crop_coords:
                original_crop_coords = [None, None, None, None, 1]
        else:
            original_crop_coords = [None, None, None, None, 1]
        return original_crop_coords

    def find_im_bb(self, crop_read_arr, nose_read_arr):
        x_min = None
        y_min = None
        x_max = None
        y_max = None
        bbox = []
        scaled_width = self.vid_width
        scaled_height = self.vid_height

        for i in range(self.num_frames):
            read_arr = crop_read_arr
            if len(read_arr) > i:
                curr_im_coords = read_arr[i]
                x_min = curr_im_coords[0] * scaled_width / 640
                y_min = curr_im_coords[2] * scaled_height / 480
                x_max = curr_im_coords[1] * scaled_width / 640
                y_max = curr_im_coords[3] * scaled_height / 480

            read_arr = nose_read_arr
            if len(read_arr) > i:
                confidence = read_arr[i][2]
                if confidence > .25:
                    x_center = read_arr[i][0]
                    y_center = read_arr[i][1]
                    norm_coords = normalize_to_camera([(x_center, y_center)], [x_min, x_max, y_min, y_max],
                                                      scaled_width=scaled_width, scaled_height=scaled_height)
                    x_center = norm_coords[0][0]
                    y_center = norm_coords[0][1]
                    bb_size = 75  # Change as desired, based on size of face

                    # We want only face, not whole body
                    x_min = int(x_center - bb_size)
                    y_min = int(y_center - bb_size)
                    x_max = int(x_center + bb_size)
                    y_max = int(y_center + bb_size)
                    x_coords = np.clip(np.array([x_min, x_max]), 0, self.vid_width)
                    y_coords = np.clip(np.array([y_min, y_max]), 0, self.vid_height)
                    x_min = x_coords[0]
                    x_max = x_coords[1]
                    y_min = y_coords[0]
                    y_max = y_coords[1]

                    bbox.append([x_min, y_min, x_max, y_max])
        return bbox

    def make_read_arr(self, f, num_constraint=None):
        read_arr = f.readlines()
        if num_constraint:
            read_arr = [read_arr[i].split(',')[0:num_constraint] for i in range(0, len(read_arr), self.fps_fraction)]
        else:
            read_arr = [read_arr[i].split(',') for i in range(0, len(read_arr), self.fps_fraction)]
        for index, num in enumerate(read_arr):
            for val_index, val in enumerate(num):
                read_arr[index][val_index] = read_arr[index][val_index].replace('(', '')
                read_arr[index][val_index] = read_arr[index][val_index].replace(')', '')
        read_arr = [[float(k) for k in i] for i in read_arr]
        return read_arr

    def write_arr(self, arr, name, extra=False):
        with open(os.path.join(self.im_dir, (name + '.txt')), mode='w') as f:
            for element in arr:
                f.write(str(element) + '\n')
            if extra:
                f.write('Rescaling factor: ' + '\n' + str(self.resize_factor) + '\n')


def normalize_to_camera(coords, crop_coord, scaled_width, scaled_height):
    if sum(crop_coord) <= 0:
        rescale_factor = (scaled_width / 256, scaled_height / 256)  # Original size was 256
    else:
        rescale_factor = ((crop_coord[1] - crop_coord[0]) / 256.0, (crop_coord[3] - crop_coord[2]) / 256.0)
    norm_coords = [
        np.array((coord[0] * rescale_factor[0] + crop_coord[0], coord[1] * rescale_factor[1] + crop_coord[2]))
        for coord in coords]
    return np.array(norm_coords)


def probe(vid_file_path):
    ''' Give a json from ffprobe command line

    :param vid_file_path : The absolute (full) path of the video file
    :type vid_file_path : str
    '''
    if type(vid_file_path) != str:
        raise Exception('Give ffprobe a full file path of the video')

    command = ["ffprobe",
               "-loglevel", "quiet",
               "-print_format", "json",
               "-show_format",
               "-show_streams",
               vid_file_path
               ]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = pipe.communicate()
    out = out.decode()
    return json.loads(out)
