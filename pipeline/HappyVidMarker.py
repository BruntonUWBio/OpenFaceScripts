import json
import sys

import os

import dill
from sklearn.externals import joblib

OpenDir = sys.argv[sys.argv.index('-d') + 1]
os.chdir(OpenDir)
scores_file = 'au_emotes.txt'
scores = json.load(open(scores_file))
classifier = joblib.load('happy_trained_RandomForest.pkl')
out_file = open('happy_vids.txt', 'w')
currMax = 0
for vid in scores:
    emotion_data = [item for sublist in
                    [b for b in [[a for a in scores[vid].values() if a]] if b]
                    for item in sublist if item[1] in ['Happy', 'Neutral', 'Sleeping']]
    au_data = []
    target_data = []
    if emotion_data:
        aus_list = sorted([int(x) for x in emotion_data[0][0].keys()])
        for frame in emotion_data:
            aus = frame[0]
            au_data.append([float(aus[str(x)]) for x in aus_list])
        predicted = classifier.predict(au_data)
        num_happy = len([x for x in predicted if x])
        if num_happy > currMax:
            out_file.write(vid + '\t' + str(num_happy))
            out_file.flush()
            currMax = num_happy
            print(vid + '\t' + str(num_happy))