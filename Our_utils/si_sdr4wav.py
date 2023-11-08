import sys
sys.path.append("/home/dsi/moradim/Audio-Visual-separation-using-RTF/") #todo- general path
from scipy.io.wavfile import read
from torch import Tensor
import torch

#script for single sample si-sdr checking
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
    _check_same_shape(preds, target)
    EPS = torch.finfo(preds.dtype).eps
    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + EPS) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + EPS)
    target_scaled = alpha * target
    noise = target_scaled - preds
    val = (torch.sum(target_scaled ** 2, dim=-1) + EPS) / (torch.sum(noise ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)
    return val

# Output signal
est_path = "/home/dsi/moradim/Audio-Visual-separation-using-RTF/Our_utils/spk1_griffin_librosa_init.wav" #todo- general path
# Reference signal
target_path = "/home/dsi/moradim/Audio-Visual-separation-using-RTF/Our_utils/clean_1.wav" #todo- general path
# load wav file
_, est = read(est_path)
_, speakers_target = read(target_path)
# From wav file to tensor
est = torch.tensor(est, dtype=torch.float32)
speakers_target = torch.tensor(speakers_target, dtype=torch.float32)
# sisdr calculation
si_sdr_value = scale_invariant_signal_distortion_ratio(est, speakers_target, zero_mean=True)
print(si_sdr_value)