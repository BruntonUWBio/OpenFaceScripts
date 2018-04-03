from scipy import signal, stats
import numpy as np
import soundfile as sf
import sys

NOISE_PERCENTILE = 96
NUM_FREQS = 3

fname = sys.argv[1]
out_fname = sys.argv[2]

audio, fs = sf.read(fname)

f, t, S = signal.stft(audio, fs=fs, nperseg=2000) # noverlap = nperseg/2

f_good = (f >= 370) & (f <= 900)

power_all = np.sum(np.square(np.abs(S[1:, :])), axis=0)
power_speech = np.sum(np.square(np.abs(S[f_good, :])), axis=0)
power_ratio = power_speech / power_all

noisy_times = power_ratio < np.percentile(power_ratio, 5)
noise = np.mean(np.abs(S[:, noisy_times]), axis=1)

ratio = noise/np.max(noise)
mult = ratio > np.percentile(ratio, NOISE_PERCENTILE)
mult = signal.convolve(mult, np.ones(NUM_FREQS), 'same')
mult[mult > 1] = 1

S_denoised = S * (1-mult[:, np.newaxis])
t, audio_denoised = signal.istft(S_denoised, fs=fs)

sf.write(out_fname, audio_denoised, fs)
