import glob
import os
import subprocess
import sys

from OpenFaceScripts import crop_image_sequence, OpenFaceScorer


class VideoImageCropper:
    def __init__(self):
        if '-v' in sys.argv:
            vid = sys.argv[sys.argv.index('-v') + 1]
        self.crop_path = sys.argv[sys.argv.index('-c') + 1]
        self.nose_path = sys.argv[sys.argv.index('-n') + 1]
        self.csv_path = sys.argv[sys.argv.index('-csv') + 1]
        self.already_cropped = ('-ac' in sys.argv)
        self.already_detected = ('-ad' in sys.argv)
        if '-id' in sys.argv:
            self.im_dir = sys.argv[sys.argv.index('-id') + 1]
        else:
            self.im_dir = os.path.splitext(vid)[0] + '_cropped'
        if not self.already_cropped and not self.already_detected:
            self.crop_txt_files = self.find_txt_files(self.crop_path)
            self.nose_txt_files = self.find_txt_files(self.nose_path)
            if not os.path.lexists(self.im_dir):
                os.mkdir(self.im_dir)
                subprocess.Popen('ffmpeg -i "{0}" -vf fps=30 "{1}"'.format(vid, os.path.join(self.im_dir, (
                    os.path.basename(vid) + '_out%04d.png'))), shell=True).wait()
            crop_image_sequence.CropImages(self.im_dir, self.crop_txt_files, self.nose_txt_files,
                                           save=True)
        if not self.already_detected:
            self.run_open_face()
        OpenFaceScorer.OpenFaceScorer(self.im_dir, self.csv_path)

    @staticmethod
    def find_txt_files(path):
        return {os.path.splitext(os.path.basename(v))[0]: v for v in
                glob.iglob(os.path.join(path + '/**/*.txt'), recursive=True)}

    def run_open_face(self):
        executable = '/home/gvelchuru/OpenFace/build/bin/FeatureExtraction' #Change to location of OpenFace
        subprocess.Popen("ffmpeg -r 30 -f image2 -s 1920x1080 -pattern_type glob -i '{0}' -b:v 2000k {1}".format(
            os.path.join(self.im_dir, '*.png'),
            os.path.join(self.im_dir,
                         'inter_out.mp4')), shell=True).wait()

        # Remove q if visualization desired, inserted for performance
        subprocess.Popen(
            '{0} -f {1} -ov {2} -of {3} -verbose -wild -q -multi-view 1'.format(executable,
                                                                                os.path.join(self.im_dir,
                                                                                             'inter_out.mp4'),
                                                                                os.path.join(self.im_dir, 'out.mp4'),
                                                                                os.path.join(self.im_dir, 'au.txt')),
            shell=True).wait()


if __name__ == '__main__':
    crop = VideoImageCropper()
