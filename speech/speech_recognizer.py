import sys
import os
import shutil
import json

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from OpenFaceScripts.runners import CropAndOpenFace
from OpenFaceScripts.scoring import AUScorer
from OpenFaceScripts.runners import SecondRunOpenFace


def applySlidingWindow(predicDic: dict) -> dict:
    startFrame = 0
    endFrame = duration(vid) * 30


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
            SecondRunOpenFace.do_second_run(
                os.path.dirname(os.path.abspath(working_directory)))
            presences = json.load(
                open(os.path.join(working_directory, 'all_dict.txt')))

            for frame in presences:
                au25_c = 1 if '25' in presences[frame] else 0
                au26_c = 1 if '26' in presences[frame] else 0
                au25_r = presences[frame]['25'] if au25_c else 'N/A'
                au26_r = presences[frame]['26'] if au26_c else 'N/A'
                confidence = presences[frame]['confidence']

                if confidence >= .95 and ((au25_c == 1 and au25_r >= 1) or
                                          (au26_c == 1 and au26_r >= 1)):
                    predicDic[frame] = True
                else:
                    predicDic[frame] = False

                # out.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(
                # frame, au25_c, au26_c, au25_r, au26_r, confidence,
                # prediction))

            predicDic = applySlidingWindow(predicDic)

            for frame in predicDic:
                out.write(frame + '\t' + predicDic[frame])

        except FileNotFoundError as e:
            print('{0} vid cannot be processed!'.format(vid))
        shutil.rmtree(working_directory)
