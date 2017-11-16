import glob
import json
import os
import sys

sys.path.append('/home/gvelchuru')
from OpenFaceScripts.runners import CropAndOpenFace, VidCropper


class CropThread:
    def __init__(self, vids):
        self.vids = vids
        self.vid_left = int(sys.argv[sys.argv.index('-vl') + 1])
        self.vid_right = int(sys.argv[sys.argv.index('-vr') + 1])
        self.run()

    def run(self):
        for vid in range(self.vid_left, self.vid_right):
            self.crop_image(vid)

    def crop_image(self, i):
        vid = self.vids[i]
        im_dir = os.path.splitext(vid)[0] + '_cropped'
        try:
            if not os.path.exists(im_dir) or 'ran_open_face.txt' not in os.listdir(im_dir):
                VidCropper.duration(vid)
                CropAndOpenFace.VideoImageCropper(vid=vid, im_dir=im_dir,
                                                  crop_txt_files=crop_txt_files, nose_txt_files=nose_txt_files,
                                                  vid_mode=True)
                open(os.path.join(im_dir, 'ran_open_face.txt'), 'w').write('Yes')
        except Exception as e:
            print(e)


if __name__ == '__main__':
    path = sys.argv[sys.argv.index('-id') + 1]

    crop_file = os.path.join(path, 'crop_files_list.txt')
    nose_file = os.path.join(path, 'nose_files_list.txt')

    # else:
    crop_txt_files = json.load(open(crop_file))
    nose_txt_files = json.load(open(nose_file))

    os.chdir(path)
    vids = [os.path.join(path, x) for x in glob.glob('*.avi')]
    CropThread(vids)
