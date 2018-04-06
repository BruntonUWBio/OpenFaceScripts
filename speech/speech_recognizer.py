import sys
import os
import shutil

import math

sys.path.append('/home/jeffery')
sys.path.append('/home/gvelchuru')
from OpenFaceScripts.runners import CropAndOpenFace
from OpenFaceScripts.runners.VidCropper import duration
from OpenFaceScripts.scoring import AUScorer

if __name__ == '__main__':
    vid = sys.argv[sys.argv.index('-v') + 1]
    out_file = sys.argv[sys.argv.index('-t') + 1]
    predicDic = {}
    with open(out_file, 'w') as out:
        out_parent_dir = os.path.dirname(out_file)
        working_directory = os.path.join(
            out_parent_dir, '{0}_speech_recognizer'.format(out_file))

        if not os.path.exists(working_directory):
            os.mkdir(working_directory)
            shutil.copy(vid, os.path.join(working_directory, 'inter_out.avi'))

        if 'au.csv' not in os.listdir(working_directory):
            CropAndOpenFace.run_open_face(working_directory, True)
        try:
            scorer = AUScorer.AUScorer(working_directory)
            presences = scorer.presence_dict
            out.write('frame\t' + 'au25_c\t' + 'au26_c\t' +
                      'au25_r\t' + 'au26_r\t' + 'confidence\t' + 'prediction\n')

            for frame in presences:
                au25_c = 1 if '25' in presences[frame] else 0
                au26_c = 1 if '26' in presences[frame] else 0
                au25_r = presences[frame]['25'] if au25_c else 'N/A'
                au26_r = presences[frame]['26'] if au26_c else 'N/A'
                confidence = presences[frame]['confidence']

                if au25_c or au26_c:
                    predicDic[frame] = "speaking"
                else:
                    predicDic[frame] = "not speaking"

                # out.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(
                    # frame, au25_c, au26_c, au25_r, au26_r, confidence,
                    # prediction))
            predic_frames = sorted(list(predicDic.keys()))

            for frame in predicDic:
                prev_frame = predic_frames[predic_frames.index(frame) - 1]
                if abs(prev_frame - frame) >= 5:
                    prev_frame = None
                next_frame = predic_frames[predic_frames.index(frame) + 1]


        except FileNotFoundError as e:
            print('{0} vid cannot be processed!'.format(vid))
        shutil.rmtree(working_directory)
