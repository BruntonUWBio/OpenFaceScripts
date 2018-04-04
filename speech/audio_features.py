import sys
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import _pickle as cPickle
from os import path

if __name__ == '__main__':
    audio = sys.argv[sys.argv.index('-a') + 1]
    out_dir = sys.argv[sys.argv.index('-od') + 1]
    filename = path.splitext(path.basename(audio))[0]

    rate, signal = wav.read(audio)
    mfcc_feats = mfcc(signal, samplerate=rate, winlen=1/30, nfft=512, numcep=26, winstep=1/30)
    np.save(path.join(out_dir, filename), mfcc_feats)
