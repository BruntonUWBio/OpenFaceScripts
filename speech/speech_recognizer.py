import sys
import os
import shutil
from collections import defaultdict
import math

sys.path.append('/home/jeffery')
# sys.path.append('/home/gvelchuru')
from OpenFaceScripts.runners import CropAndOpenFace
from OpenFaceScripts.runners.VidCropper import duration
from OpenFaceScripts.scoring import AUScorer

def make_mouth_presences(au_csv):
    au_file_lines = open(au_csv).readlines()
    line_one = au_file_lines[0].split(', ')

    frame_loc = 0
    au25_loc = -4
    au26_loc = -3

    d = defaultdict(list)
    for i, line in enumerate(au_file_lines):
        if i != 0:
            items = line.split(', ')
            if float(items[au25_loc]) == 1.00 or float(items[au26_loc]) == 1.00:
                d[int(items[frame_loc]) - 1].append(True)
            else:
                d[int(items[frame_loc]) - 1].append(False)

    return d


if __name__ == '__main__':
    vid = sys.argv[sys.argv.index('-v') + 1]
    out_file = sys.argv[sys.argv.index('-t') + 1]
    with open(out_file, 'w') as out:
        out_parent_dir = os.path.dirname(out_file)
        working_directory = os.path.join(out_parent_dir, '{0}_speech_recognizer'.format(out_file))
        if not os.path.exists(working_directory):  # remove for faster debugging
            os.mkdir(working_directory)
            shutil.copy(vid, os.path.join(working_directory, 'inter_out.avi'))
            CropAndOpenFace.run_open_face(working_directory, True)
        presences = make_mouth_presences(os.path.join(working_directory, 'au.csv'))
        start_frame = 0
        last_frame = None
        for frame in range(int(math.ceil(duration(vid) * 30))):
            if frame not in presences:
                temp_string = "not recognized"
            else:
                if presences[frame][0]:  # can set to specific vals as well
                    temp_string = "open mouth"
                else:
                    temp_string = "closed mouth"
            out.write(str(frame) + '\t' + temp_string + '\n')
        shutil.rmtree(working_directory)

