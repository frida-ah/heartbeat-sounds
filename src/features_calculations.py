import os
import glob

import numpy as np
import pandas as pd
import librosa as lr
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def create_dataset():
    dataset = []
    for folder in ["/data/set_a/**", "./data/set_b/**"]:
        for filename in glob.iglob(folder):
            label = os.path.basename(filename).split("_")[0]
            dataset.append({"filename": filename, "label": label})
    dataset_df = pd.DataFrame(dataset)
    music = shuffle(dataset_df, random_state=42)
    return music


def calculate_features(music):
    for label in music.label.unique():
        audio, sfreq = lr.load(music[music.label == label].filename.iloc[0], duration=4)
        D = lr.amplitude_to_db(np.abs(lr.stft(audio)), ref=np.max)
        mfccs = lr.feature.mfcc(y=audio, sr=sfreq, n_mfcc=40)
    return audio, sfreq, D, mfccs


def rolling_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    rolling_mean = (cumsum[N:] - cumsum[:-N]) / N
    return rolling_mean


def calc_smooth_sound_envelope(audio):
    """
    By calculating the envelope of each sound and smoothing it,
    I've eliminated much of the noise and have a cleaner signal
    of when a heartbeat is happening
    """
    # Rectify the audio signal
    audio_rectified = np.absolute(audio)
    # Smooth by applying a rolling mean
    audio_envelope = rolling_mean(audio_rectified, 50)
    return audio_envelope


def plot_auditory_envelope(audio_envelope):
    plt.plot(audio_envelope)
    plt.show()


def calc_features_envelope(audio_envelope):
    envelope_mean = np.mean(audio_envelope, axis=0)
    envelope_std = np.std(audio_envelope, axis=0)
    envelope_max = np.max(audio_envelope, axis=0)
    return envelope_mean, envelope_std, envelope_max


music = create_dataset()
audio, sfreq, D, mfccs = calculate_features(music)
audio_envelope = calc_smooth_sound_envelope(audio)
plot_auditory_envelope(audio_envelope)
envelope_mean, envelope_std, envelope_max = calc_features_envelope(audio_envelope)
