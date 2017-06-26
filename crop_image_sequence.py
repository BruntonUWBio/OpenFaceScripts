# Requires pose, crop coordinates
import glob
import os
import numpy as np
import cv2
import pickle
from scipy import misc


class CropImages:
    def __init__(self, directory, crop_txt_files, nose_txt_files, save=False):
        self.fps_fraction = 1
        self.crop_txt_files = crop_txt_files
        self.nose_txt_files = nose_txt_files
        save_name = None
        self.im_dir = directory
        files = glob.glob(os.path.join(self.im_dir, '*.png'))
        files = sorted(files)
        self.read_arr_dict = {}
        crop_array_name = os.path.join(directory, 'cropped.txt')
        if os.path.lexists(crop_array_name):
            with open(crop_array_name, mode='rb') as f:
                self.crop_im_arr_arr_dict = pickle.load(f)
        else:
            self.crop_im_arr_arr_dict = {image: self.make_crop_im_arr_arr(image) for image in files}
            with open(crop_array_name, mode='wb') as f:
                pickle.dump(self.crop_im_arr_arr_dict, f)
        bb_arr = [None, None, None, None]
        for image in self.crop_im_arr_arr_dict.keys():
            im_crop_arr = self.crop_im_arr_arr_dict[image]
            if im_crop_arr:
                x_min = im_crop_arr[1]
                y_min = im_crop_arr[2]
                x_max = im_crop_arr[3]
                y_max = im_crop_arr[4]
                if bb_arr[0]:
                    bb_arr = [min(bb_arr[0], x_min), min(bb_arr[1], y_min),
                              max(bb_arr[2], x_max), max(bb_arr[3], y_max)]
                else:
                    bb_arr = [x_min, y_min, x_max, y_max]

        self.write_arr(bb_arr, 'bb_arr')
        for image in self.crop_im_arr_arr_dict.keys():
            path_name, base_name = os.path.split(image)
            if 'cropped' not in base_name:
                split_name = os.path.splitext(base_name)
                if save:
                    save_name = os.path.join(path_name, split_name[0] + '_cropped' + split_name[1])
                if self.crop_im_arr_arr_dict[image] is not None:
                    img = self.crop_im_arr_arr_dict[image][0]
                    self.crop_image(img, save_name, bb_arr)
                os.remove(image)

    def make_crop_im_arr_arr(self, name):
        img = misc.imread(name, mode='RGB')
        return self.crop_predictor(img, name, scaled_height=img.shape[0], scaled_width=img.shape[1])

    def crop_image(self, img, save_name, crop_arr):
        x_min, y_min, x_max, y_max = self.return_min_max(crop_arr)
        crop_im = img[y_min:y_max, x_min:x_max]
        if save_name:
            crop_im = misc.imresize(cv2.cvtColor(crop_im, cv2.COLOR_RGB2BGR),
                                    (crop_im.shape[0] * 5, crop_im.shape[1] * 5))
            cv2.imwrite(save_name, crop_im)

    def write_arr(self, arr, name):
        with open(os.path.join(self.im_dir, (name + '.txt')), mode='w') as f:
            for element in arr:
                f.write(str(element) + '\n')

    def crop_predictor(self, img, name, scaled_width, scaled_height, make_cropped_im=False):
        print('Name: {0}'.format(name))
        base_name = os.path.basename(name)
        crop_file_path, file_num = self.find_crop_path(base_name, self.crop_txt_files)
        print('Crop file: {0}'.format(crop_file_path))
        x_min = None
        y_min = None
        x_max = None
        y_max = None
        if crop_file_path is not None:
            if crop_file_path not in self.read_arr_dict.keys():
                with open(crop_file_path) as f:
                    self.read_arr_dict[crop_file_path] = self.make_read_arr(f)
            read_arr = self.read_arr_dict[crop_file_path]
            i = file_num - 1
            if len(read_arr) > i:
                curr_im_coords = read_arr[i]
                x_min = curr_im_coords[0] * scaled_width / 640
                y_min = curr_im_coords[2] * scaled_height / 480
                x_max = curr_im_coords[1] * scaled_width / 640
                y_max = curr_im_coords[3] * scaled_height / 480

        nose_file_path, file_num = self.find_crop_path(base_name, self.nose_txt_files)
        print('Nose file: {0}'.format(nose_file_path))
        if nose_file_path is not None:
            i = file_num - 1
            if nose_file_path not in self.read_arr_dict.keys():
                with open(nose_file_path) as f:
                    self.read_arr_dict[nose_file_path] = self.make_read_arr(f)
            read_arr = self.read_arr_dict[nose_file_path]
            if len(read_arr) > i:
                confidence = read_arr[i][2]
                print('Crop Confidence: {0}'.format(confidence))
                if confidence > .25:
                    x_center = read_arr[i][0]
                    y_center = read_arr[i][1]
                    norm_coords = self.normalize_to_camera([(x_center, y_center)], [x_min, x_max, y_min, y_max],
                                                           scaled_width=scaled_width, scaled_height=scaled_height)
                    x_center = norm_coords[0][0]
                    y_center = norm_coords[0][1]
                    bb_size = 75  # Change as desired, based on size of face

                    # We want only face, not whole body
                    x_min = int(x_center - bb_size)
                    y_min = int(y_center - bb_size)
                    x_max = int(x_center + bb_size)
                    y_max = int(y_center + bb_size)
                    im = img
                    x_coords = np.clip(np.array([x_min, x_max]), 0, im.shape[0])
                    y_coords = np.clip(np.array([y_min, y_max]), 0, im.shape[1])
                    x_min = x_coords[0]
                    x_max = x_coords[1]
                    y_min = y_coords[0]
                    y_max = y_coords[1]

                    # If cropping is desired, return the cropped image - else, return the original image with bounding
                    # box for further analysis
                    if make_cropped_im:
                        crop_im = im[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]].copy()
                        return [crop_im, x_min, y_min, x_max, y_max]
                    else:
                        return [img, x_min, y_min, x_max, y_max]

    @staticmethod
    def normalize_to_camera(coords, crop_coord, scaled_width, scaled_height):
        if sum(crop_coord) <= 0:
            rescale_factor = (scaled_width / 256, scaled_height / 256)  # Original size was 256
        else:
            rescale_factor = ((crop_coord[1] - crop_coord[0]) / 256.0, (crop_coord[3] - crop_coord[2]) / 256.0)
        norm_coords = [
            np.array((coord[0] * rescale_factor[0] + crop_coord[0], coord[1] * rescale_factor[1] + crop_coord[2]))
            for coord in coords]
        return np.array(norm_coords)

    @staticmethod
    def find_crop_path(file, crop_txt_files):
        parts = file.split('.')
        pid = parts[0]
        try:
            out_num = int(''.join(parts[1][parts[1].index('out') + 3: len(parts[1])]))
        except ValueError:
            return None, None
        out_file = None
        if pid in list(crop_txt_files.keys()):
            out_file = crop_txt_files[pid]
        return out_file, out_num

    @staticmethod
    def return_min_max(arr):
        return int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])

    def make_read_arr(self, f, num_constraint=None):
        read_arr = f.readlines()
        if num_constraint is not None:
            read_arr = [read_arr[i].split(',')[0:num_constraint] for i in range(0, len(read_arr), self.fps_fraction)]
        else:
            read_arr = [read_arr[i].split(',') for i in range(0, len(read_arr), self.fps_fraction)]
        for index, num in enumerate(read_arr):
            for val_index, val in enumerate(num):
                read_arr[index][val_index] = val.replace('(', '')
                val = read_arr[index][val_index]
                read_arr[index][val_index] = val.replace(')', '')
        read_arr = [[float(k) for k in i] for i in read_arr]
        return read_arr
