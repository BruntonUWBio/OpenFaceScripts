"""
.. module SecondRunOpenFace
    :synopsis: Module for use after an initial run of OpenFace on a video set, attempts to rerun on the videos
        that OpenFace could not recognize a face in the first time.
"""
import os
import subprocess
import sys

import numpy as np


def do_second_run(patient_directory):
    os.chdir(patient_directory)
    files = [
        x for x in (os.path.join(patient_directory, vid_dir)
                    for vid_dir in os.listdir(patient_directory))
        if (os.path.isdir(x) and 'au.csv' in os.listdir(x))
    ]

    num_GPUs = 2
    processes = []
    indices = list(map(int, np.linspace(0, len(files), num=num_GPUs + 1)))

    for index in range(len(indices) - 1):
        processes.append(
            subprocess.Popen(
                [
                    'python3',
                    '/home/gvelchuru/OpenFaceScripts/helpers/SecondRunHelper.py',
                    '-od', patient_directory, '-vl',
                    str(indices[index]), '-vr',
                    str(indices[index + 1])
                ],
                env={'CUDA_VISIBLE_DEVICES': '{0}'.format(str(index))}))
    [p.wait() for p in processes]


if __name__ == '__main__':
    patient_directory = sys.argv[sys.argv.index('-od') + 1]
    do_second_run(patient_directory)
