import sys
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

if __name__ == '__main__':
    audio = sys.argv[sys.argv.index('-a') + 1]
    out_file = sys.argv[sys.argv.index('-t') + 1]
    with open(out_file, 'w') as out:
        signal, rate = wav.read(audio)
        mfcc_feats = mfcc(signal, rate, winlen=1/30, nfft=512, numcep=26, winstep=1/30)
        np.savetxt(out, mfcc_feats, delimiter='\t')
