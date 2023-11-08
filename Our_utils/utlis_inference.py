import numpy as np  
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
import os

def plot_spectrogram(masks, title, save_path, ylabel='freq_bin', aspect='auto', xmax=None):
    masks  =masks.cpu()
    for indx_mask in range(masks.shape[1]):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(f"Spectrogram (db) - {title}")
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        im = axs.imshow(masks[0, indx_mask, :, :].detach().numpy(), origin='lower', aspect=aspect) #sample 0 from batch
        if xmax:
            axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_path).joinpath(f"Mask_Speaker_{indx_mask}"))
        plt.close('all')
        
def save_audio(mix_waves, separated_signals, save_path, bit16):
    samplerate = 16000
    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    mix_waves = mix_waves[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
    Path(save_path).mkdir(parents=True, exist_ok=True)
    if bit16 == 16:
        write(os.path.join(save_path, "Mixed_0.wav"), samplerate, mix_waves.astype(np.float16))
        write(os.path.join(save_path, "Speaker_0.wav"), samplerate, separated_audio1.astype(np.float16))
        write(os.path.join(save_path, "Speaker_1.wav"), samplerate, separated_audio2.astype(np.float16))
    else:
        write(os.path.join(save_path, "Mixed_0.wav"), samplerate, mix_waves.astype(np.float32))
        write(os.path.join(save_path, "Speaker_0.wav"), samplerate, separated_audio1.astype(np.float32))
        write(os.path.join(save_path, "Speaker_1.wav"), samplerate, separated_audio2.astype(np.float32))
        
def save_vad(vad_output, save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vad_output = vad_output.cpu()
    for spk in range(vad_output.shape[1]):
        est_vad = torch.where(vad_output[0, spk] >= 0.5, 1, 0)
        plt.plot(est_vad)
        plt.savefig(os.path.join(save_path, f"estimated_vad_{spk}.png"))
        plt.close()