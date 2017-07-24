"""
.. module MultiCropper
    :synopsis: Script to apply cropping and OpenFace to all videos in a directory.

"""

import glob
import json
import os
import subprocess
import sys
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import CropAndOpenFace


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
    raise ValueError('I found no duration')
    # return None

def probe(vid_file_path):
    ''' Give a json from ffprobe command line

    @vid_file_path : The absolute (full) path of the video file, string.
    '''
    if type(vid_file_path) != str:
        raise Exception('Gvie ffprobe a full file path of the video')
        return

    command = ["ffprobe",
               "-loglevel", "quiet",
               "-print_format", "json",
               "-show_format",
               "-show_streams",
               vid_file_path
               ]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = pipe.communicate()
    out = out.decode("utf-8")
    return json.loads(out)


def crop_image(i):
    vid = vids[i]
    im_dir = os.path.splitext(vid)[0] + '_cropped'
    try:
        duration(vid)
        CropAndOpenFace.VideoImageCropper(vid=vid, im_dir=im_dir, crop_path=crop_path, nose_path=nose_path,
                                      crop_txt_files=crop_txt_files, nose_txt_files=nose_txt_files, vid_mode=True)
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    path = sys.argv[sys.argv.index('-id') + 1]
    crop_path = sys.argv[sys.argv.index('-c') + 1]
    nose_path = sys.argv[sys.argv.index('-n') + 1]

    crop_file = os.path.join(path, 'crop_files_list.txt')
    nose_file = os.path.join(path, 'nose_files_list.txt')

    if not os.path.exists(crop_file):
        crop_txt_files = CropAndOpenFace.find_txt_files(crop_path)
        json.dump(crop_txt_files, open(crop_file, mode='w'))
    else:
        crop_txt_files = json.load(open(crop_file, mode='r'))
    if not os.path.exists(nose_file):
        nose_txt_files = CropAndOpenFace.find_txt_files(nose_path)
        json.dump(nose_txt_files, open(nose_file, mode='w'))
    else:
        nose_txt_files = json.load(open(nose_file, mode='r'))

    os.chdir(path)
    processes = []
    vids = sorted(glob.glob('*.avi'))
    vids = [os.path.join(path, x) for x in vids]
    multiProcessingNum = 2 #Number of GPUs, adjust as necessary

    p = Pool(multiProcessingNum)
    p.map(crop_image, range(len(vids)))
    p.close()

    # for vid in vids:
    #     if 'aa97abcd_4_00' in vid:
    #         crop_image(vids.index(vid))