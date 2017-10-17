import glob
import json
import sys

import os

emotions = ['Neutral', 'Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised']
au_list = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]
OpenDir = sys.argv[sys.argv.index('-od') + 1]
ck_dir = sys.argv[sys.argv.index('-ck') + 1]
ck_dict = {}
facs_dir = os.path.join(ck_dir, 'FACS')
for facs_file in glob.glob(os.path.join(facs_dir, '**/*.txt'), recursive=True):
    basename = os.path.basename(facs_file)
    basename = basename.replace('_facs.txt', '')
    ck_dict[basename] = {}
    with open(facs_file) as file:
        au_dict = {}
        for line in file.readlines():
            line = [x for x in line.strip().split(' ') if x]
            if line:
                line = list(map(complex, line))
                line = [c.real for c in line]
                au_dict[line[0]] = line[1]
        popList = []
        for num in au_dict:
            if int(num) not in au_list:
                popList.append(num)
        for num in popList:
            au_dict.pop(num)
    for num in au_list:
        if float(num) not in au_dict:
            au_dict[float(num)] = 0.0
        elif au_dict[float(num)] == 0:
            au_dict[float(num)] = 1.0
    ck_dict[basename] = [au_dict]
emotion_dir = os.path.join(ck_dir, 'Emotion')
for emotion_file in glob.glob(os.path.join(emotion_dir, '**/*.txt'), recursive=True):
    basename = os.path.basename(emotion_file)
    basename = basename.replace('_emotion.txt', '')
    with open(emotion_file) as file:
        line = file.read()
        line = [x for x in line.strip().split() if x]
        line = complex(line[0]).real
        ck_dict[basename].append(emotions[int(line)])
for name in ck_dict:
    if len(ck_dict[name]) == 1:
        ck_dict[name].append(None)
os.chdir(OpenDir)
json.dump(ck_dict, open('ck_dict.txt', 'w'))