import sys
import os
import shutil
import json
from typing import List
import numpy as np

sys.path.append('/home/jeffery/')
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from OpenFaceScripts.runners import CropAndOpenFace
from OpenFaceScripts.scoring import AUScorer
from OpenFaceScripts.runners import SecondRunOpenFace


def auditok_oscillating_predictions(predicDic: dict, vid: str,
                                    auditok_file: str) -> dict:
    FRAMES_PER_STATE_THRESH = 10
    out_dict = {}
    with open(auditok_file) as audi_data:
        for line in audi_data.readlines():
            audi_line_info = line.split()
            starting_frame = int(int(audi_line_info[1]) / 30)
            ending_frame = int(int(audi_line_info[2]) / 30)
            open_sections = []  # type: List[List[int]]
            closed_sections = []  # type: List[List[int]]
            start_frame = 0
            curr_string = None

            for frame in range(starting_frame, ending_frame + 1):
                if frame not in presences:
                    temp_string = None
                else:
                    if predicDic[frame]:
                        temp_string = True
                    else:
                        temp_string = False

                if curr_string is None or curr_string != temp_string:
                    if temp_string is not None:
                        if temp_string:
                            open_sections.append([start_frame, frame - 1])
                        else:
                            closed_sections.append([start_frame, frame - 1])
                    start_frame = frame
                    curr_string = temp_string

            average_open_fps = np.average([x[1] - x[0] for x in open_sections])
            average_close_fps = np.average(
                [x[1] - x[0] for x in closed_sections])

            for frame in range(starting_frame, ending_frame + 1):
                if frame in predicDic:
                    if np.average(
                            average_open_fps,
                            average_close_fps) <= FRAMES_PER_STATE_THRESH:
                        out_dict[frame] = "speaking"
                    else:
                        out_dict[frame] = "not speaking"

            return out_dict


if __name__ == '__main__':
    VID = sys.argv[sys.argv.index('-v') + 1]
    OUT_FILE = sys.argv[sys.argv.index('-t') + 1]
    AUDI_FILE = sys.argv[sys.argv.index('-a') + 1]
    PREDIC_DIC = {}
    with open(OUT_FILE, 'w') as out:
        OUT_PARENT_DIR = os.path.dirname(OUT_FILE)
        WORKING_DIR = os.path.join(OUT_PARENT_DIR,
                                   '{0}_speech_recognizer'.format(OUT_FILE))

        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)
            shutil.copy(VID, os.path.join(WORKING_DIR, 'inter_out.avi'))

        if 'au.csv' not in os.listdir(WORKING_DIR):
            CropAndOpenFace.run_open_face(WORKING_DIR, True)
        
        try:
            SecondRunOpenFace.do_second_run(
                os.path.dirname(os.path.abspath(WORKING_DIR)))
            presences = json.load(
                open(os.path.join(WORKING_DIR, 'all_dict.txt')))
            
            for frame in presences:
                au25_c = 1 if '25' in presences[frame] else 0
                au26_c = 1 if '26' in presences[frame] else 0
                au25_r = presences[frame]['25'] if au25_c else 'N/A'
                au26_r = presences[frame]['26'] if au26_c else 'N/A'
                confidence = presences[frame]['confidence']

                PREDIC_DIC[frame] = bool(confidence >= .95
                                         and ((au25_c == 1 and au25_r >= 1) or
                                              (au26_c == 1 and au26_r >= 1)))

                predicDic = auditok_oscillating_predictions(PREDIC_DIC, VID, AUDI_FILE)

            for frame in predicDic:
                out.write(frame + '\t' + predicDic[frame])

        except FileNotFoundError as e:
            print('{0} vid cannot be processed!'.format(VID))
        # shutil.rmtree(WORKING_DIR)
