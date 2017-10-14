#need to run this separately
"""
===========
MovieWriter
===========
This example uses a MovieWriter directly to grab individual frames and write
them to a file. This avoids any event loop integration, but has the advantage
of working with even the Agg backend. This is not recommended for use in an
interactive setting.
"""
# -*- noplot -*-
import random

import numpy as np
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import sys
import os

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=1, metadata=metadata)

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# times = np.loadtxt("times_100_2.txt")
# corr = np.loadtxt("corr_100_2.txt")
vid_dir = '/data2/OpenFaceTests/cb46fd46_7_0135_cropped'
vid = '/data2/OpenFaceTests/cb46fd46_7_0135.avi'
test_line = os.path.join(vid_dir, 'writer_test_line.mp4')

times = [_ for _ in range(600)]
corr = [random.uniform(0, 1) for _ in range(600)]
norm = plt.Normalize()
colors = plt.cm.jet(norm(corr))

fig = plt.figure(figsize=(14,2))
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
        sys.stdout.write('{:.2f} {}/{}\r'.format(100*i/len(times), i, len(times)))
        sys.stdout.flush()

        low = max(i-50, 0)
        high = min(i+51, len(times)-1)
        # plt.xlim(times[low], times[high])
        plt.ylim(0.35, 0.75)
        vl.set_segments([[[times[i], 0],[times[i],1]]])
        # ax.set_axis_bgcolor(colors[i])
        # li.set_ydata(corr[i-10:i+11])
        # li.set_xdata(times[i-10:i+11])
        fig.canvas.draw()
        # plt.pause(0.001)
        writer.grab_frame()

subprocess.Popen('ffmpeg -y -i {0} -vf "movie={1}, '
                 'scale=640:-1, format=argb, colorchannelmixer=aa=.75 [inner]; [in][inner] overlay=.269 [out]" '
                 '-strict -2 {2}'.format(vid, test_line, os.path.join(vid_dir, 'completed.mp4')), shell=True).wait()