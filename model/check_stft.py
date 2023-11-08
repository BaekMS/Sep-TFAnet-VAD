
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from torchaudio import transforms
import pickle
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import librosa


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(f"Spectrogram (db) - {title}")
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(spec, origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  fig.savefig(title + '.png')



n_fftBins = 512
sample_rate = 16000

am_to_db = transforms.AmplitudeToDB(stype="power")
spec = transforms.Spectrogram(n_fft=n_fftBins, hop_length=n_fftBins//2, win_length=n_fftBins,
                                      window_fn=torch.hann_window, power=None, )  # for all channels
inv_spec = transforms.InverseSpectrogram(n_fft=n_fftBins, hop_length=n_fftBins//2, win_length=n_fftBins,
                                      window_fn=torch.hann_window)
resample = transforms.Resample(16000, 16000/4)
recording_df = pd.read_csv("/mnt/dsi_vol1/shared/sharon_db/mordehay/train/csv_files/with_white_noise_res.csv")
record_path = recording_df.loc[0, "path_file"]

example = True

if example:
    wav_fname = "/dsi/gannot-lab/datasets/LibriSpeech/LibriSpeech/Train/8193/116804/8193-116804-0045.wav"
    samplerate, data = read(wav_fname)
    if data.dtype is not np.float32:
        data = data.astype(np.float32)
    max = 1
    min = -1
    data = 2*max * (data - data.min()) / (data.max() - data.min()) + min

    lib_stft = librosa.stft(data, n_fft=512, hop_length=512//2, win_length=512)
    lib_power = torch.pow(torch.abs(torch.tensor(lib_stft)), 2)
    lib_spectrum = librosa.power_to_db(lib_power)
    plot_spectrogram(lib_spectrum, title="example_audio_lib")

    #data = 2*(data - data.min()) / (data.max() - data.min()) - 1
    signal = np.expand_dims(data, axis=0)
    signal_ten = torch.tensor(signal)
    stft = spec(signal_ten)
    power = torch.pow(torch.abs(stft[0,:,:]), 2)
    spectrum = am_to_db(power)
    plot_spectrogram(spectrum, title="example_audio")

if not example:
    with open(record_path, "rb") as f:
        mixed_sig_np, speakers_target, s_thetas_array = pickle.load(f)
    #clean = np.sum(speakers_target, axis=0, keepdims=True)
    clean = np.expand_dims(speakers_target[0], axis=0)
    clean_norm = 2 * (clean - clean.min()) / (clean.max() - clean.min()) - 1
    mixed_audio = np.expand_dims(mixed_sig_np[0], axis=0)#first channel

    for name, signal in zip(["clean", "mixed"], [clean_norm, mixed_audio]):

        write(f"original_{name}.wav", sample_rate, signal.T)
        signal_ten = torch.tensor(signal)
        #signal_ten = resample(signal_ten)
        stft = spec(signal_ten)
        power = torch.pow(torch.abs(stft[0,:,:]), 2)
        spectrum = am_to_db(power)
        plot_spectrogram(spectrum, title=name)

        reco_audio= inv_spec(stft[0, :, :], length=signal.shape[1]) #first channel
        write(f"reco_{name}.wav", sample_rate, reco_audio.numpy().T)

# %%
