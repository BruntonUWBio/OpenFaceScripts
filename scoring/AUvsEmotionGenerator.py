import functools
import glob
import json
import multiprocessing
import os
import sys
from collections import defaultdict
from os.path import join
from dask import dataframe as df
import progressbar
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from OpenFaceScripts.helpers.SecondRunHelper import process_eyebrows, get_vid_from_dir
from OpenFaceScripts import AUGui
from OpenFaceScripts.scoring import AUScorer
from OpenFaceScripts.runners import VidCropper
from OpenFaceScripts.helpers.patient_info import patient_day_session, get_patient_names
from tqdm import tqdm


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
        au_frame = df.read_hdf(
            os.path.join('all_' + patient, '*.hdf'), '/data')

        # Restrict to a subset of frames which are relevant
        au_frame = au_frame[au_frame.patient == patient and au_frame.day == day
                            and au_frame.session == session]

        csv_path = join(patient_dir,
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

                    if curr_au_frame:
                        to_write = clean_to_write(csv_dict[i])
                        curr_au_frame['annotated'] = to_write
    except FileNotFoundError as not_found_error:
        print(not_found_error)


def find_one_patient_scores(patient_dirs: List[str], patient: str):
    """Finds the annotated emotions for a single patient and adds to overall patient DataFrame.
    :param patient_dirs: All directories ran through OpenFace.
    :param patient: Patient to find annotated emotions for
    """
    tqdm_position, patient = patient

    for patient_dir in tqdm(patient_dirs, position=tqdm_position):
        if patient in patient_dir:
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
    PARTIAL_FIND_FUNC = functools.partial(find_scores)
    TUPLE_PATIENTS = [((i % cpu_count()), x) for i, x in enumerate(PATIENTS)]
    Pool().map(PARTIAL_FIND_FUNC, TUPLE_PATIENTS)
