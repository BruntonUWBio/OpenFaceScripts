"""
.. module MultiCropper
    :synopsis: Script to apply cropping and OpenFace to all videos in a directory.

"""

import glob
import json
import os
import subprocess
import sys
import numpy as np

from OpenFaceScripts import CropAndOpenFace

sys.path.append('home/gvelchuru/')

path = sys.argv[sys.argv.index('-id') + 1]
crop_path = sys.argv[sys.argv.index('-c') + 1]
nose_path = sys.argv[sys.argv.index('-n') + 1]
crop_txt_files = CropAndOpenFace.find_txt_files(crop_path)
nose_txt_files = CropAndOpenFace.find_txt_files(nose_path)

crop_file = os.path.join(path, 'crop_files_list.txt')
nose_file = os.path.join(path, 'nose_files_list.txt')

if not os.path.exists(crop_file):
    json.dump(crop_txt_files, open(crop_file, mode='w'))
if not os.path.exists(nose_file):
    json.dump(nose_txt_files, open(nose_file, mode='w'))


os.chdir(path)
processes = []
vids = sorted(glob.glob('*.avi'))
multiProcessingNum = 6
for i in range(0, len(vids), multiProcessingNum): #Two GPUs, adjust as necessary
    processes = [subprocess.Popen(
        'python3 /home/gvelchuru/OpenFaceScripts/CropAndOpenFace.py -v {0} -c {1} -n {2}'.format(vids[i + x], crop_path,
                                                                                                 nose_path), shell=True) for x in range(multiProcessingNum) if i+x in range(len(vids))]
    [p.wait() for p in processes]

#
# for vid in vids:
#     im_dir = os.path.splitext(vid)[0] + '_cropped'
#     CropAndOpenFace.VideoImageCropper(vid, im_dir, crop_path=crop_path, nose_path=nose_path, crop_txt_files=crop_txt_files, nose_txt_files=nose_path)