from tqdm import tqdm
from typing import List
import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from OpenFaceScripts.helpers.patient_info import patient_day_session, get_patient_names
from OpenFaceScripts.runners import VidCropper
from OpenFaceScripts.scoring import AUScorer
from OpenFaceScripts import AUGui
from OpenFaceScripts.helpers.SecondRunHelper import process_eyebrows, get_vid_from_dir
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from dask import dataframe as df
from os.path import join
from collections import defaultdict
import multiprocessing
import json
import glob
import functools
"""
.. module:: AUvsEmotionGenerator
    :synopsis: Appends classified emotion to AU dataframe
"""


def clean_to_write(to_write: str) -> str:
    if to_write == 'Surprised':
        to_write = 'Surprise'
    elif to_write == 'Disgusted':
        to_write = 'Disgust'
    elif to_write == 'Afraid':
        to_write = 'Fear'

    return to_write


def find_scores(patient_dir: str):
    """
    Finds the scores for a specific patient directory

    :param patient_dir: Directory to look in
    """
    try:
        patient, day, session = patient_day_session(patient_dir)
        # au_frame = df.read_hdf(
        # os.path.join('all_' + patient, 'au_*.hdf'), '/data')
        # try:
        au_frame = df.read_hdf(
            os.path.join(patient_dir, 'hdfs', 'au_0.hdf'), '/data')
        # except ValueError as e:
            # print(e)

            # return

        if 'annotated' in au_frame.columns:
            return

        # Restrict to a subset of frames which are relevant
        # au_frame = au_frame[au_frame.patient == patient and au_frame.session == day
        # and au_frame.vid == session]
        au_frame = au_frame[au_frame.patient == patient]
        # try:
        au_frame = au_frame[au_frame.session == day]
        # except AttributeError as e:
            # print(e)

            # return
        au_frame = au_frame[au_frame.vid == session]

        csv_path = join(
            patient_dir,
            os.path.basename(patient_dir).replace('_cropped', '') +
            '_emotions.csv')
        num_frames = int(
            VidCropper.duration(get_vid_from_dir(patient_dir)) * 30)

        if os.path.exists(csv_path):
            csv_dict = AUGui.csv_emotion_reader(csv_path)

            if csv_dict:
                annotated_ratio = int(num_frames / len(csv_dict))

                if annotated_ratio == 0:
                    annotated_ratio = 1
                csv_dict = {
                    i * annotated_ratio: c

                    for i, c in csv_dict.items()
                }

                for i in [
                        x for x in csv_dict.keys() if 'None' not in csv_dict[x]
                ]:
                    curr_au_frame = au_frame[au_frame.frame == i]

                    if not curr_au_frame.empty:
                        print('hi!')
                        to_write = clean_to_write(csv_dict[i])
                        curr_au_frame['annotated'] = to_write
        df.to_hdf(
            au_frame,
            os.path.join(patient_dir, 'hdfs', 'au_*.hdf'),
            '/data',
            format='table')
    except FileNotFoundError as not_found_error:
        print(not_found_error)


def find_one_patient_scores(patient_dirs: List[str], patient: tuple):
    """Finds the annotated emotions for a single patient and adds to overall patient DataFrame.

    :param patient_dirs: All directories ran through OpenFace.
    :param patient: Patient to find annotated emotions for
    """
    tqdm_position, patient = patient
    curr_dirs = [x for x in patient_dirs if patient in x]

    for patient_dir in tqdm(curr_dirs, position=tqdm_position):
        find_scores(patient_dir)


if __name__ == '__main__':
    OPEN_DIR = sys.argv[sys.argv.index('-d') + 1]
    os.chdir(OPEN_DIR)
    # Directories have been previously cropped by CropAndOpenFace
    PATIENT_DIRS = [
        x for x in glob.glob('*cropped') if 'hdfs' in os.listdir(x)
    ]
    PATIENTS = get_patient_names(PATIENT_DIRS)
    # EYEBROW_DICT = process_eyebrows(OPEN_DIR,
    # open(join(OPEN_DIR, 'eyebrows.txt')))
    # PARTIAL_FIND_FUNC = functools.partial(find_one_patient_scores, PATIENT_DIRS)
    # TUPLE_PATIENTS = [((i % cpu_count()), x) for i, x in enumerate(PATIENTS)]
    # Pool().map(PARTIAL_FIND_FUNC, TUPLE_PATIENTS)
    # Pool().map(find_scores, PATIENTS)

    for i, x in enumerate(PATIENTS):
        tuple_patient = (i % cpu_count(), x)
        find_one_patient_scores(PATIENT_DIRS, tuple_patient)
