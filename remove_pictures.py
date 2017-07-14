import glob
import os

#Script used for cleaning up a directory of pictures
os.chdir('/home/gvelchuru/cb46fd46')
for pic in glob.iglob('*.png'):
    print(pic)
    os.remove(pic)