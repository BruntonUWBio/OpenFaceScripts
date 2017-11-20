import glob
import json
import os
import sys

import progressbar
from OpenFaceScripts.runners import CropAndOpenFace, VidCropper
from helpers.HalfCropper import try_cropping

if __name__ == '__main__':
    path = sys.argv[sys.argv.index('-id') + 1]

    crop_file = os.path.join(path, 'crop_files_list.txt')
    nose_file = os.path.join(path, 'nose_files_list.txt')

    if not os.path.exists(crop_file):
        crop_path = sys.argv[sys.argv.index('-c') + 1]
        crop_txt_files = CropAndOpenFace.find_txt_files(crop_path)
        json.dump(crop_txt_files, open(crop_file, mode='w'))
    else:
        crop_txt_files = json.load(open(crop_file))
    if not os.path.exists(nose_file):
        nose_path = sys.argv[sys.argv.index('-n') + 1]
        nose_txt_files = CropAndOpenFace.find_txt_files(nose_path)
        json.dump(nose_txt_files, open(nose_file, mode='w'))
    else:
        nose_txt_files = json.load(open(nose_file))

    os.chdir(path)
    vids = [os.path.join(path, x) for x in glob.glob('*.avi')]
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(vids))
    for index, vid in enumerate(vids):
        bar.update(index)
        im_dir = os.path.splitext(vid)[0] + '_cropped'
        try_cropping(vid, im_dir)
