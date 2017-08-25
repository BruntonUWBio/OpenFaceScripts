"""
.. module MultiCropper
    :synopsis: Script to apply cropping and OpenFace to all videos in a directory.

"""

import glob
import json
import os
import sys

import progressbar
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts.runners import CropAndOpenFace, VidCropper


def crop_image(i):
    vid = vids[i]
    im_dir = os.path.splitext(vid)[0] + '_cropped'
    try:
        if not os.path.exists(im_dir) or 'au.txt' not in os.listdir(im_dir):
            VidCropper.duration(vid)
            CropAndOpenFace.VideoImageCropper(vid=vid, im_dir=im_dir,
                                              crop_txt_files=crop_txt_files, nose_txt_files=nose_txt_files,
                                              vid_mode=True)
    except Exception as e:
        print(e)


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
    processes = []
    vids = [os.path.join(path, x) for x in glob.glob('*.avi')]
    multiProcessingNum = 2  # 2 GPUs

    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=1)
    for i, _ in enumerate(Pool(multiProcessingNum).imap(crop_image, range(len(vids)), chunksize=10), 1):
        bar.update(i / len(vids))
