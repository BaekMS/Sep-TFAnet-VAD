import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write


def plot_spectrogram_masks(masks, title, save_path, ylabel='freq_bin', aspect='auto', xmax=None):
    masks  =masks.detach().cpu()
    for indx_mask in range(masks.shape[1]):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(f"{title}")
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        im = axs.imshow(masks[0, indx_mask, :, :], origin='lower', aspect=aspect) #sample 0 from batch
        if xmax:
            axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
        Path(f"{save_path}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}/mask_{indx_mask}")
        plt.close('all')
    
def plot_spectrogram(masks, title, save_path, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(f"{title}")
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(masks, origin='lower', aspect=aspect) #sample 0 from batch
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    Path(f"{save_path}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_path}/MixSpec{title}.png")
    plt.close('all')
        
def save_audio(mix_waves, separated_signals, target, save_path, samplerate, mean_sisdr=None):
    target_audio1 = target[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    target_audio2 = target[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    mix_waves = mix_waves[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
    Path(save_path).mkdir(parents=True, exist_ok=True)
    write(f"{save_path}/mixed.wav", samplerate, mix_waves.astype(np.float32))
    write(f"{save_path}/output_0.wav", samplerate, separated_audio1.astype(np.float32))
    write(f"{save_path}/output_1.wav", samplerate, separated_audio2.astype(np.float32))
    write(f"{save_path}/clean_0.wav", samplerate, target_audio1.astype(np.float32)) ########
    write(f"{save_path}/clean_1.wav", samplerate, target_audio2.astype(np.float32)) ##########
    if mean_sisdr is not None:
        f = open(f"{save_path}//The Si-Sdr is {mean_sisdr:.2f}.txt", "w")

def save_vad(vad_output, target_vad, save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    #print(vad_output.shape)
    #print(target_vad.shape)
    vad_output = vad_output.cpu()
    target_vad = target_vad.cpu()
    for spk in range(target_vad.shape[1]):
        plt.plot(target_vad[0, spk])
        plt.savefig(f"{save_path}/True_Vad_{spk}.png")
        plt.close()
        est_vad = torch.where(vad_output[0, spk] >= 0.7, 1, 0)
        plt.plot(est_vad)
        plt.savefig(f"{save_path}/Estimated_Vad_{spk}.png")
        plt.close()

    
def calc_vad(masks):
    """
    mask with shaoe of [B, num_spk, F, T]
    """
    threshold_masks = torch.where(masks >= 0.5, 1, 0)
    output_simple_vad = torch.where(torch.sum(threshold_masks, dim=2) >= 257*0.25, 1, 0)
    return output_simple_vad