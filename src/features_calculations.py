import os
import glob
from typing import Tuple

import numpy as np
import pandas as pd
import librosa as lr
import matplotlib.pyplot as plt
from librosa.core import stft, amplitude_to_db
from sklearn.utils import shuffle
from librosa.display import specshow


def create_dataset() -> pd.DataFrame:
    dataset = []
    for folder in ["/data/set_a/**", "./data/set_b/**"]:
        for filename in glob.iglob(folder):
            label = os.path.basename(filename).split("_")[0]
            dataset.append({"filename": filename, "label": label})
    dataset_df = pd.DataFrame(dataset)
    music = shuffle(dataset_df, random_state=42)
    return music


def calculate_features(
    music: pd.DataFrame,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    for label in music.label.unique():
        audio, sfreq = lr.load(music[music.label == label].filename.iloc[0], duration=4)
        D = lr.amplitude_to_db(np.abs(lr.stft(audio)), ref=np.max)
        mfccs = lr.feature.mfcc(y=audio, sr=sfreq, n_mfcc=40)
    return (audio, sfreq, D, mfccs)


def rolling_mean(x: int, N: int) -> np.ndarray:
    cumsum = np.cumsum(np.insert(x, 0, 0))
    rolling_mean = (cumsum[N:] - cumsum[:-N]) / N
    return rolling_mean


def calc_smooth_sound_envelope(audio: np.ndarray) -> np.ndarray:
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


def plot_auditory_envelope(audio_envelope: np.ndarray) -> None:
    plt.plot(audio_envelope)
    plt.show()


def calc_features_envelope(audio_envelope: np.ndarray) -> Tuple[np.float32, np.float32, np.float32]:
    envelope_mean = np.mean(audio_envelope, axis=0)
    envelope_std = np.std(audio_envelope, axis=0)
    envelope_max = np.max(audio_envelope, axis=0)
    return (envelope_mean, envelope_std, envelope_max)


def calc_tempos(audio: np.ndarray, sfreq: int) -> np.ndarray:
    tempos = []
    audio_df = pd.DataFrame(data=audio)

    for _col, i_audio in audio_df.items():
        tempos.append(lr.beat.tempo(i_audio.values, sr=sfreq, hop_length=2 ** 6, aggregate=None))

    tempos = np.array(tempos)
    return tempos


def calc_tempo_stats(tempos: np.ndarray) -> Tuple[np.float32, np.float32, np.float32]:
    tempos_mean = tempos.mean(axis=-1)
    tempos_std = tempos.std(axis=-1)
    tempos_max = tempos.max(axis=-1)
    return (tempos_mean, tempos_std, tempos_max)


def fourier_transformation(audio: np.ndarray) -> np.ndarray:
    spec = stft(audio, hop_length=2 ** 4, n_fft=2 ** 7)
    spec_db = amplitude_to_db(np.abs(spec))  # convert into decibels
    return spec_db


def plot_fourier_transformation(sfreq: int, audio: np.ndarray, spec_db: np.ndarray) -> None:
    # Compare the raw audio to the spectrogram of the audio
    time = np.arange(0, len(audio)) / sfreq
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(time, audio)
    specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=2 ** 4)
    plt.show()


def calc_bandwidths_centroids(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bandwidths = lr.feature.spectral_bandwidth(audio)[0]
    centroids = lr.feature.spectral_centroid(audio)[0]
    return (bandwidths, centroids)


music = create_dataset()
(audio, sfreq, D, mfccs) = calculate_features(music)
audio_envelope = calc_smooth_sound_envelope(audio)
plot_auditory_envelope(audio_envelope)
(envelope_mean, envelope_std, envelope_max) = calc_features_envelope(audio_envelope)
tempos = calc_tempos(audio, sfreq)
(tempos_mean, tempos_std, tempos_max) = calc_tempo_stats(tempos)
spec = stft(audio, hop_length=2 ** 4, n_fft=2 ** 7)
spec_db = fourier_transformation(audio)
plot_fourier_transformation(sfreq, audio, spec_db)
(bandwidths, centroids) = calc_bandwidths_centroids(audio)
