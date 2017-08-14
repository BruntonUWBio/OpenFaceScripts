# crops videos and sends to openface script
import glob
import os
import subprocess
import sys

from OpenFaceScripts import crop_image_sequence
from runners import CropAndOpenFace

vid_dir = sys.argv[(sys.argv.index('-vd')) + 1]
os.chdir(vid_dir)
avis = sorted(glob.glob(os.path.join(vid_dir, '**/*.avi'), recursive=True))
outFile = os.path.join(vid_dir, os.path.basename(vid_dir))

# outString = 'avimerge -o {0}.avi -i '.format(outFile)
# outString += ' {0}  {1}'.format(avis[0], avis[1])
# subprocess.Popen(outString, shell=True).wait()
# old_outfile = copy.deepcopy(outFile)
# new_outfile = None

if not os.path.lexists(outFile):
    os.mkdir(outFile)

processes = []
crop_txt_files = find_txt_files('/data2/storagedrive/crop_coords')
nose_txt_files = find_txt_files('/data2/storagedrive/pose_coords')
for index, avi in enumerate(avis):
    if 1 < index < 200:  # Change based on how many videos wanted
        out_png = os.path.join(outFile, os.path.basename(avi) + '_out%04d.png')
        processes.append(subprocess.Popen('ffmpeg -i "{0}" -vf fps=30 "{1}"'.format(avi, out_png), shell=True))
[p.wait() for p in processes]
crop_image_sequence.CropImages(outFile, crop_txt_files, nose_txt_files,
                               save=True)
CropAndOpenFace.run_open_face(outFile)
