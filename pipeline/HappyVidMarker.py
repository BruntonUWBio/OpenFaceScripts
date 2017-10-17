import functools
import glob
import json
import os
import random
import sys

import cv2
import multiprocessing

import progressbar
from OpenFaceScripts.runners.SecondRunOpenFace import get_vid_from_dir
import subprocess

from OpenFaceScripts.runners.VidCropper import duration
from sklearn.externals import joblib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as manimation

from pathos.multiprocessing import ProcessingPool as Pool

OpenDir = sys.argv[sys.argv.index('-d') + 1]
os.chdir(OpenDir)


def bar_movie(vid, vid_dir, times, corr):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=len(times)/duration(vid), metadata=metadata)

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    # times = np.loadtxt("times_100_2.txt")
    # corr = np.loadtxt("corr_100_2.txt")
    test_line = os.path.join(vid_dir, 'writer_test_line.mp4')

    norm = plt.Normalize()
    colors = plt.cm.jet(norm(corr))

    fig = plt.figure(figsize=(14, 2))
    ax = fig.add_subplot(111)

    # data = [0 for i in range(21)]
    # data[0] = 0
    # data[1] = 1

    # some X and Y data
    # x = np.arange(len(data))
    # y = data

    li, = ax.plot(times, corr)
    vl = ax.vlines(0, 0, 1)

    with writer.saving(fig, test_line, 100):
        for i in range(len(times)):
            # for i in range(100):
            sys.stdout.write('{:.2f} {}/{}\r'.format(100 * i / len(times), i, len(times)))
            sys.stdout.flush()

            low = max(i - 50, 0)
            high = min(i + 51, len(times) - 1)
            # plt.xlim(times[low], times[high])
            # plt.ylim(0.35, 0.75)
            vl.set_segments([[[times[i], 0], [times[i], 1]]])
            # ax.set_axis_bgcolor(colors[i])
            # li.set_ydata(corr[i-10:i+11])
            # li.set_xdata(times[i-10:i+11])
            fig.canvas.draw()
            # plt.pause(0.001)
            writer.grab_frame()

    subprocess.Popen('ffmpeg -y -i {0} -vf "movie={1}, '
                     'scale=640:-1, format=argb, colorchannelmixer=aa=.75 [inner]; [in][inner] overlay=.269 [out]" '
                     '-strict -2 {2}'.format(vid, test_line, os.path.join(vid_dir, 'completed.mp4')), shell=True).wait()

    plt.close()

scores_file = 'au_emotes.txt'
scores = json.load(open(scores_file))
classifier = joblib.load('happy_trained_RandomForest.pkl')
# for vid in scores:
#     emotion_data = [item for sublist in
#                     [b for b in [[a for a in scores[vid].values() if a]] if b]
#                     for item in sublist]
#     au_data = []
#     if emotion_data:
#         aus_list = sorted([int(x) for x in emotion_data[0][0].keys()])
#         for frame in emotion_data:
#             aus = frame[0]
#             au_data.append([float(aus[str(x)]) for x in aus_list])
#         predicted = classifier.predict(au_data)
#         num_happy = len([x for x in predicted if x])
#         if num_happy > 2000:
#             out_file.write(vid + '\t' + str(num_happy))
#             out_file.flush()
#             print(vid + '\t' + str(num_happy) + '\n')

# out_file = open('happy_vids.txt', 'w')

def mark_vid_dir(out_q, vid_dir):

    # vid_dir = os.path.join(OpenDir, 'cb46fd46_7_0135_cropped')
    vid = get_vid_from_dir(vid_dir)
    # subprocess.Popen('ffmpeg -y -i "{0}" -vf fps=30 "{1}"'.format(vid, os.path.join(vid_dir, (
    #                 os.path.basename(vid) + '_out%04d.png'))), shell=True).wait()
    # images = glob.glob(os.path.join(vid_dir, '*.png'))
    # emotion_data = [item for sublist in
    #                 [b for b in [[a for a in scores[os.path.basename(vid_dir)].values() if a]] if b]
    #                 for item in sublist]

    emotion_data = []
    scores_dict = scores[os.path.basename(vid_dir)]
    for frame in range(int(duration(vid) * 30)):
        if str(frame) in scores_dict:
            emotion_data.append(scores_dict[str(frame)])
        else:
            emotion_data.append(None)


    # au_data = []
    aus_list = sorted([1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45])
    predicted_arr = []
    # emotes = []
    for frame in emotion_data:
        if frame:
            aus = frame[0]
            au_data = ([float(aus[str(x)]) for x in aus_list])
            predicted = classifier.predict_proba(np.array(au_data).reshape(1, -1))[0]
            predicted_arr.append(predicted)
            # emotes.append(frame[1])
        else:
            predicted_arr.append(np.array([0, 0]))

    times = [x for x in range(0, len(predicted_arr), 10)]
    corr = [predicted_arr[a][1] for a in times]
    bar_movie(vid, vid_dir, times, corr)

    # for index, image_name in enumerate(sorted(images)):
    #     if index in range(len(predicted)) and predicted[index]:
    #         image = cv2.imread(image_name)
    #         cv2.rectangle(image, (0, 0), (40, 40), (0, 255, 0), -1)
    #         cv2.imwrite(image_name, image)
    # subprocess.Popen("ffmpeg -y -r 30 -f image2 -pattern_type glob -i '{0}' -b:v 7000k {1}".format(
    #             os.path.join(vid_dir, '*.png'),
    #             os.path.join(vid_dir,
    #                          'marked.mp4')), shell=True).wait()
    out_q.put({vid_dir: sum(corr)/len(corr)})

vids_file = 'happy_predic_vids.txt'
vids_done = {}
original_len = len(vids_done)
files = [x for x in (os.path.join(OpenDir, vid_dir) for vid_dir in os.listdir(OpenDir)) if
         (os.path.isdir(x) and 'au.txt' in os.listdir(x))]
if os.path.exists(vids_file):
    vids_done = json.load(vids_file)
remaining = [x for x in files if x not in vids_done]
out_q = multiprocessing.Manager().Queue()
f = functools.partial(mark_vid_dir, out_q)
bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(remaining))
for i, _ in enumerate(Pool().imap(f, remaining), 1):
    bar.update(i)
while not out_q.empty():
    vids_done.update(out_q.get())
if len(vids_done) != original_len:
    json.dump(vids_done, open(scores_file, 'w'))