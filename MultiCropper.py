import glob
import os
import subprocess
import sys

sys.path.append('home/gvelchuru/')
from OpenFaceScripts import CropAndOpenFace

path = sys.argv[sys.argv.index('-id') + 1]
crop_path = sys.argv[sys.argv.index('-c') + 1]
nose_path = sys.argv[sys.argv.index('-n') + 1]

os.chdir(path)
processes = []
vids = sorted(glob.glob('*.avi'))
multiProcessingNum = 8
for i in range(0, len(vids), multiProcessingNum): #Two GPUs, adjust as necessary
    processes = [subprocess.Popen(
        'python3 /home/gvelchuru/OpenFaceScripts/CropAndOpenFace.py -v {0} -c {1} -n {2}'.format(vids[i + x], crop_path,
                                                                                                 nose_path), shell=True) for x in range(multiProcessingNum) if i+x in range(len(vids))]
    [p.wait() for p in processes]
    #vid = vids[i]
    #im_dir = os.path.splitext(vid)[0] + '_cropped'
    #CropAndOpenFace.VideoImageCropper(vids[i], im_dir, crop_path=crop_path, nose_path=nose_path)
