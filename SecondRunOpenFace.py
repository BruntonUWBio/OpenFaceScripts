"""
.. module SecondRunOpenFace
    :synopsis: Module for use after an initial run of OpenFace on a video set, attempts to rerun on the videos
        that OpenFace could not recognize a face in the first time.
"""
import glob
import os
import shutil
import sys
import subprocess

import cv2
import functools
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import AUScorer, CropAndOpenFace


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


class OpenFaceSecondRunner:
    """
    Main runner class
    """

    def __init__(self, directory):
        vid_dirs = (os.path.join(directory, vid_dir) for vid_dir in os.listdir(directory) if
                    os.path.isdir(os.path.join(directory, vid_dir)))
        with open(os.path.join(directory, 'hsv.txt'), mode='w') as log:
            for vid_dir in vid_dirs:
                if 'au.txt' in os.listdir(vid_dir):
                    scorer = AUScorer.AUScorer(vid_dir, 0, False)
                    if not all(scorer.emotions.values()):
                        vid = os.path.join(vid_dir, 'out.avi')
                        hsv_changed_dir = os.path.join(os.path.dirname(vid), 'hsv_changed')
                        if not os.path.exists(hsv_changed_dir):
                            os.mkdir(hsv_changed_dir)
                        subprocess.Popen(
                            'ffmpeg -y -i "{0}" -q:v 2 -vf fps=30 "{1}"'.format(vid, os.path.join(hsv_changed_dir, (
                                os.path.basename(vid) + '_out%04d.png'))), shell=True).wait()
                        p = Pool()
                        pngs = [os.path.join(hsv_changed_dir, x) for x in os.listdir(hsv_changed_dir) if '.png' in x]
                        f = functools.partial(make_more_bright, pngs)
                        p.map(f, range(len(pngs)))
                        CropAndOpenFace.run_open_face(hsv_changed_dir)
                        new_scorer = AUScorer.AUScorer(hsv_changed_dir)
                        if all(new_scorer.emotions.values()):
                            log.write(vid_dir + 'has been recognized! \n')
                            log.flush()
                        else:
                            log.write('No change for ' + vid_dir + '\n')
                            log.flush()

if __name__ == '__main__':
    directory = sys.argv[sys.argv.index('-od') + 1]
    OpenFaceSecondRunner(directory)
