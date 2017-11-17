"""
.. module MultiCropper
    :synopsis: Script to apply cropping and OpenFace to all videos in a directory.

"""

import glob
import json
import os
import subprocess
import sys

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts.runners import CropAndOpenFace

if __name__ == '__main__':

    path = sys.argv[sys.argv.index('-id') + 1]

    crop_file = os.path.join(path, 'crop_files_list.txt')
    nose_file = os.path.join(path, 'nose_files_list.txt')

    if not os.path.exists(crop_file):
        crop_path = sys.argv[sys.argv.index('-c') + 1]
        crop_txt_files = CropAndOpenFace.find_txt_files(crop_path)
        json.dump(crop_txt_files, open(crop_file, mode='w'))

    if not os.path.exists(nose_file):
        nose_path = sys.argv[sys.argv.index('-n') + 1]
        nose_txt_files = CropAndOpenFace.find_txt_files(nose_path)
        json.dump(nose_txt_files, open(nose_file, mode='w'))

    folder_components = os.listdir(path)
    vids = [x for x in glob.glob(os.path.join(path, '*.avi')) if (
    os.path.splitext(x)[0] + '_cropped' not in folder_components or 'ran_open_face.txt' not in os.listdir(
        os.path.join(path, os.path.splitext(x)[0] + '_cropped')))]

    mid = len(vids) // 2
    p1 = subprocess.Popen(
        ['python3', '/home/gvelchuru/OpenFaceScripts/helpers/HalfCropper.py', '-id', path, '-vl', str(0), '-vr',
         str(mid)],
        env={'CUDA_VISIBLE_DEVICES': '0'})
    p2 = subprocess.Popen(
        ['python3', '/home/gvelchuru/OpenFaceScripts/helpers/HalfCropper.py', '-id', path, '-vl', str(mid), '-vr',
         str(len(vids))],
        env={'CUDA_VISIBLE_DEVICES': '1'})
    p1.wait()
    p2.wait()
