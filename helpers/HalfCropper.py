import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runners.MultiCropAndOpenFace import make_vids, make_crop_and_nose_files
from runners import CropAndOpenFace, VidCropper
from runners.VidCropper import DurationException
from tqdm import tqdm


def run():
    for vid in tqdm(range(vid_left, vid_right + 1)):
        crop_image(vid)


def crop_image(i):
    vid = vids[i]
    im_dir = os.path.splitext(vid)[0] + '_cropped'
    try_cropping(vid, im_dir)


def try_cropping(vid, im_dir):
    try:
        VidCropper.duration(vid)
        CropAndOpenFace.VideoImageCropper(
            vid=vid,
            im_dir=im_dir,
            crop_txt_files=crop_txt_files,
            nose_txt_files=nose_txt_files,
            vid_mode=True)
    except DurationException as e:
        print(str(e) + '\t' + vid)


if __name__ == '__main__':
    path = sys.argv[sys.argv.index('-id') + 1]

    crop_txt_files, nose_txt_files = make_crop_and_nose_files(path)

    os.chdir(path)
    vids = make_vids(path)
    vid_left = int(sys.argv[sys.argv.index('-vl') + 1])
    vid_right = int(sys.argv[sys.argv.index('-vr') + 1])
    run()
