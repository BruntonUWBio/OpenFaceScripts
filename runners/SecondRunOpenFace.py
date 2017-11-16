"""
.. module SecondRunOpenFace
    :synopsis: Module for use after an initial run of OpenFace on a video set, attempts to rerun on the videos
        that OpenFace could not recognize a face in the first time.
"""
import copy
import functools
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import progressbar
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts.scoring import AUScorer
from OpenFaceScripts.runners import CropAndOpenFace, VidCropper




if __name__ == '__main__':


    multiprocessingNum = 2  # 2 GPUs

    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(files))
    # for i, _ in enumerate(Pool(multiprocessingNum).imap(f, files, chunksize=100), 1):
    #     bar.update(i)
    # for i, vid in enumerate(files, 1):


    json.dump(eyebrow_dict, open(os.path.join(patient_directory, 'eyebrow_dict.txt'), 'w'))
    # json.dump(already_ran, open(second_runner_files, 'w'), indent='\t')
