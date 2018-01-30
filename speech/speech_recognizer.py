import sys
import os
import shutil

import math

sys.path.append('/home/gvelchuru')
from OpenFaceScripts.runners import CropAndOpenFace
from OpenFaceScripts.runners.VidCropper import duration
from OpenFaceScripts.scoring import AUScorer

if __name__ == '__main__':
    vid = sys.argv[sys.argv.index('-v') + 1]
    out_file = sys.argv[sys.argv.index('-t') + 1]
    with open(out_file, 'w') as out:
        out_parent_dir = os.path.dirname(out_file)
        working_directory = os.path.join(out_parent_dir, 'temp_speech_recognizer')
        if not os.path.exists(working_directory):
            os.mkdir(working_directory)
            shutil.copy(vid, os.path.join(working_directory, 'inter_out.avi'))
            CropAndOpenFace.run_open_face(working_directory, True)
        presences = AUScorer.AUScorer(working_directory).presence_dict
        start_frame = 0
        curr_string = None
        for frame in range(int(math.ceil(duration(vid) * 30))):
            if frame not in presences:
                temp_string = "not recognized"
            else:
                if '25' in presences[frame] or '26' in presences[frame]:  # can set to specific vals as well
                    temp_string = "open mouth"
                else:
                    temp_string = "closed mouth"
            if not curr_string or curr_string != temp_string:
                if curr_string:
                    out.write(str(start_frame) + '\t' + str(frame - 1) + '\t' + curr_string + '\n')
                start_frame = frame
                curr_string = temp_string
        shutil.rmtree(working_directory)
