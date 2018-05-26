"""
.. module SecondRunOpenFace
    :synopsis: Module for use after an initial run of OpenFace on a video set, attempts to rerun on the videos
        that OpenFace could not recognize a face in the first time.
"""
import os
import subprocess
import sys
from collections import defaultdict
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
    # TODO: Separate by patient
    patient_map = set()

    for vid_dir in files:
        patient = vid_dir.split('_')[0]
        patient_map.add(patient)

    # indices = list(map(int, np.linspace(0, len(files), num=num_GPUs + 1)))
    split_patients = np.array_split(list(patient_map), 2)

    for index in range(2):
        curr_split_patients = split_patients[index]
        cli_command = [
            'python3',
            '/home/gvelchuru/OpenFaceScripts/helpers/SecondRunHelper.py',
            '-od',
            patient_directory,
            '--',
        ]

        for patient in curr_split_patients:
            cli_command.append(patient)
        processes.append(
            subprocess.Popen(
                cli_command,
                env={'CUDA_VISIBLE_DEVICES': '{0}'.format(str(index))}))
    [p.wait() for p in processes]


if __name__ == '__main__':
    patient_directory = sys.argv[sys.argv.index('-od') + 1]
    do_second_run(patient_directory)
