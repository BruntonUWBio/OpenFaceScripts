# crops videos and sends to openface script
import glob
import os
import subprocess
import sys

from runners import CropAndOpenFace
from runners.CropAndOpenFace import find_txt_files

vid_dir = sys.argv[(sys.argv.index('-vd')) + 1]
os.chdir(vid_dir)
avis = sorted(glob.glob(os.path.join(vid_dir, '**/*.avi'), recursive=True))
outFile = os.path.join(vid_dir, os.path.basename(vid_dir))

if not os.path.lexists(outFile):
    os.mkdir(outFile)

processes = []
crop_txt_files = find_txt_files('/data2/unclaimed/storagedrive/crop_coords')
nose_txt_files = find_txt_files('/data2/unclaimed/storagedrive/pose_coords')

for index, avi in enumerate(avis):
    if 1 < index < 200:  # Change based on how many videos wanted
        out_png = os.path.join(outFile, os.path.basename(avi) + '_out%04d.png')
        processes.append(subprocess.Popen('ffmpeg -i "{0}" -vf fps=30 "{1}"'.format(avi, out_png), shell=True))
[p.wait() for p in processes]
crop_image_sequence.CropImages(outFile, crop_txt_files, nose_txt_files,
                               save=True)
CropAndOpenFace.run_open_face(outFile)
