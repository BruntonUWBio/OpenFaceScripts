import glob
import os
import sys
import subprocess

sys.path.append("/home/gvelchuru/")
from OpenFaceScripts import crop_image


class VideoImageCropper:
    def __init__(self):
        vid = sys.argv[sys.argv.index('-v') + 1]
        self.crop_path = sys.argv[sys.argv.index('-c') + 1]
        self.nose_path = sys.argv[sys.argv.index('-n') + 1]
        self.crop_txt_files = self.find_txt_files(self.crop_path)
        self.nose_txt_files = self.find_txt_files(self.nose_path)
        self.im_dir = os.path.splitext(vid)[0] + '_cropped'
        if not os.path.lexists(self.im_dir):
            os.mkdir(self.im_dir)
            subprocess.Popen('ffmpeg -i "{0}" -vf fps=30 "{1}"'.format(vid, os.path.join(self.im_dir, (
                os.path.basename(vid) + '_out%04d.png'))), shell=True).wait()
        for image in glob.glob(os.path.join(self.im_dir, '*.png')):
            path_name, base_name = os.path.split(image)
            split_name = os.path.splitext(base_name)
            crop_image.CropImage(image, self.crop_txt_files, self.nose_txt_files,
                                 saveName=os.path.join(path_name, split_name[0] + '_cropped' + split_name[1]))
            os.remove(image)

    @staticmethod
    def find_txt_files(path):
        return {os.path.splitext(os.path.basename(v))[0]: v for v in
                glob.iglob(os.path.join(path + '/**/*.txt'), recursive=True)}


if __name__ == '__main__':
    crop = VideoImageCropper()
