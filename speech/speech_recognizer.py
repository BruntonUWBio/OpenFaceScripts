import sys
import os
import shutil

import math

sys.path.append('/home/gvelchuru')
from OpenFaceScripts.runners import CropAndOpenFace
from OpenFaceScripts.runners.VidCropper import duration
from OpenFaceScripts.scoring import AUScorer


def curr_lip_pos(mouth_points, moves, frame):
    # ret_list = []
    # try:
    #     for point in mouth_points:
    #         if 'x_{0}'.format(point) in moves[frame]:
    #             ret_list.append((moves[frame]['x_{0}'.format(point)], moves[frame]['y_{0}'.format(point)]))
    #     return ret_list
    # except KeyError as e:
    #     print('kurwa')
    return [(moves[frame]['x_{0}'.format(point)], moves[frame]['y_{0}'.format(point)]) for point in mouth_points
            if 'x_{0}'.format(point) in moves[frame]]


def aspect_ratio(curr_lip_coords):
    min_x = min([x[0] for x in curr_lip_coords])
    min_y = min([x[1] for x in curr_lip_coords])
    max_x = max([x[0] for x in curr_lip_coords])
    max_y = max([x[1] for x in curr_lip_coords])
    return (max_y - min_y) / (max_x - min_x)


if __name__ == '__main__':
    vid = sys.argv[sys.argv.index('-v') + 1]
    out_file = sys.argv[sys.argv.index('-t') + 1]
    with open(out_file, 'w') as out:
        out_parent_dir = os.path.dirname(out_file)
        working_directory = os.path.join(out_parent_dir, '{0}_speech_recognizer'.format(out_file))
        if not os.path.exists(working_directory):  # remove for faster debugging
            # shutil.rmtree(working_directory)
            os.mkdir(working_directory)
            shutil.copy(vid, os.path.join(working_directory, 'inter_out.avi'))
            CropAndOpenFace.run_open_face(working_directory, True)
        scorer = AUScorer.AUScorer(working_directory)
        moves = scorer.x_y_dict
        # presences = AUScorer.AUScorer(working_directory).presence_dict
        start_frame = 0
        curr_string = None
        mouth_points = list(range(49, 69))
        last_frame = None
        for frame in range(int(math.ceil(duration(vid) * 30))):
            if frame not in moves:
                temp_string = "not recognized"
            else:
                # if '25' in presences[frame] or '26' in presences[frame]:  # can set to specific vals as well
                #     temp_string = "open mouth"
                # else:
                #     temp_string = "closed mouth"
                if last_frame:
                    curr_lip_coords = curr_lip_pos(mouth_points, moves, frame)
                    curr_aspect = aspect_ratio(curr_lip_coords)
                    prev_lip_coords = curr_lip_pos(mouth_points, moves, last_frame)
                    prev_aspect = aspect_ratio(prev_lip_coords)
                    tol = 10 ** -2  # Change for  performance
                    if abs(prev_aspect - curr_aspect) >= tol:
                        temp_string = "open mouth"
                    else:
                        temp_string = "closed mouth"
                else:
                    temp_string = "no previous data"
                last_frame = frame
            if not curr_string or curr_string != temp_string:
                if curr_string:
                    out.write(str(start_frame) + '\t' + str(frame - 1) + '\t' + curr_string + '\n')
                start_frame = frame
                curr_string = temp_string
