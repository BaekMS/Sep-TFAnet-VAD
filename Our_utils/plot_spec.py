import pickle
import sys
sys.path.append("/home/dsi/moradim/Audio-Visual-separation-using-RTF/") #todo- general path
from pathlib import Path
import matplotlib.pyplot as plt
from torchaudio import transforms
import torch
from scipy.io.wavfile import read, write
from torch import Tensor

#script for spectrogram visualization
mel_bool = False

def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError("Predictions and targets are expected to have the same shape")

def scale_invariant_signal_distortion_ratio(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
    """Calculates Scale-invariant signal-to-distortion ratio (SI-SDR) metric. The SI-SDR value is in general
    considered an overall measure of how good a source sound.
    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        zero_mean:
            If to zero mean target and preds or not
    Returns:
        si-sdr value of shape [...]
    Example:
        #>>> from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        #>>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        #>>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        #>>> scale_invariant_signal_distortion_ratio(preds, target)
        tensor(18.4030)
    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.
    """
    #print(f"shape preds: {preds.shape} \nshape target: {target.shape}")
    _check_same_shape(preds, target)
    EPS = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + EPS) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + EPS
    )
    target_scaled = alpha * target

    noise = target_scaled - preds
    val = (torch.sum(target_scaled ** 2, dim=-1) + EPS) / (torch.sum(noise ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)
    return val

def plot_spectrogram(masks, title, save_path, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(f"Spectrogram (db) - {title}")
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(masks, origin='lower', aspect=aspect) #sample 0 from batch
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    Path(f"{save_path}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_path}/spec.png")
    plt.close('all')

with open(r"/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/test/"
          r"with_wham_noise_audio/scenario_118.p", "rb") as f: #todo- general path

    #load pickle with wav file
    mix_without_noise, noisy_signal, _, speakers_target, s_thetas_array = pickle.load(f)
#Or directly load wav files #todo- general path
_, noisy_signal = read("/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/"
                       "Separation_oldDisginedtrainer5_minOverSpk_Test/"
                       "Batch_18_SiSDRI_5.05_SiSDR_3.09_Reverb_0.36_Snr_6.16/output_0.wav")
_, clean_signal = read("/dsi/gannot-lab/datasets/mordehay/Result/test_for_git/"
                       "Separation_oldDisginedtrainer5_minOverSpk_Test/"
                       "Batch_18_SiSDRI_5.05_SiSDR_3.09_Reverb_0.36_Snr_6.16/clean_0.wav")

mel_bool= False #for mel_spec input
if mel_bool:    
    save_path = "/home/dsi/moradim/" #todo- general path
    am_to_db = transforms.AmplitudeToDB(stype="power")
    mel_spec = transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512, hop_length=256, n_mels=128)
    inv_scale = transforms.InverseMelScale(n_mels=128, sample_rate=16000, n_stft=1028, tolerance_change=1e-05)
    mel_mix = mel_spec(torch.tensor(speakers_target[0]))
    mel_mix[:,:3] = 0
    #reg_mix = inv_scale(mel_mix)
    #reg_mix = librosa.feature.inverse.mel_to_stft(np.array(mel_mix))
    power = torch.pow(torch.abs(mel_mix), 2)
    spectrum = am_to_db(power)
    plot_spectrogram(spectrum, "Spec", save_path)
    
else:
    save_path = "/home/dsi/moradim/" #todo- general path
    option = 'withNormAndZeroMean'
    am_to_db = transforms.AmplitudeToDB(stype="power")
    spec = transforms.Spectrogram(n_fft=512, hop_length=256, win_length=512,
                                      window_fn=torch.hann_window, power=None)
    noisy_signal = torch.tensor(noisy_signal, dtype=torch.float32)
    #normalization
    #noisy_signal = noisy_signal - torch.mean(noisy_signal)
    noisy_signal = 2*(noisy_signal - torch.min(noisy_signal, dim=-1, keepdim=True)[0]) / \
                   (torch.max(noisy_signal, dim=-1, keepdim=True)[0] -
                    torch.min(noisy_signal, dim=-1, keepdim=True)[0]) - 1
    spec_mix = spec(noisy_signal)
    spec_mix[0,:] = 0
    power_noisy = torch.pow(torch.abs(spec_mix), 2)
    spectrum_noisy = am_to_db(power_noisy)
    
    #for clean
    clean_signal = torch.tensor(clean_signal, dtype=torch.float32)
    spec_mix = spec(clean_signal)
    spec_mix[0] = 0
    power_clean = torch.pow(torch.abs(spec_mix), 2)
    spectrum_clean = am_to_db(power_clean)

    plot_spectrogram(power_clean, "Spec", save_path + 'clean')
    plot_spectrogram(power_noisy, "Spec", save_path + 'noisy')
    F, T = spectrum_noisy.shape
    si_sdr_value = scale_invariant_signal_distortion_ratio(noisy_signal, clean_signal, zero_mean=True)
    print(f"The sisdr is {si_sdr_value}")