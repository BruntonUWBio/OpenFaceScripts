import json
import os
import sys
import numpy as np
import matplotlib
from OpenFaceScripts.runners.VidCropper import duration
from runners.SecondRunOpenFace import get_vid_from_dir
from sklearn.externals import joblib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

OpenDir = sys.argv[sys.argv.index('-d') + 1]
os.chdir(OpenDir)

vid_dir = 'cb46fd46_7_0135_cropped'
vid = get_vid_from_dir(vid_dir)
# subprocess.Popen('ffmpeg -y -i "{0}" -vf fps=30 "{1}"'.format(vid, os.path.join(vid_dir, (
#                 os.path.basename(vid) + '_out%04d.png'))), shell=True).wait()
# images = glob.glob(os.path.join(vid_dir, '*.png'))
# emotion_data = [item for sublist in
#                 [b for b in [[a for a in scores[os.path.basename(vid_dir)].values() if a]] if b]
#                 for item in sublist]

emotion_data = []
scores_file = 'au_emotes.txt'
scores = json.load(open(scores_file))
scores_dict = scores[os.path.basename(vid_dir)]
classifier = joblib.load('Happy_trained_RandomForest.pkl')

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
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=len(times)/duration(vid), metadata=metadata)

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
plt.savefig('sample_trace.png')