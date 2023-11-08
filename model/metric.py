from turtle import forward
import torch
from torch import Tensor
try:
    from torchmetrics import PIT as PIT
    
except:
    from torchmetrics import PermutationInvariantTraining as PIT 
import sys
#print(sys.path)
#from  pit_wrapper import PITLossWrapper
#from sdr import pairwise_neg_sisdr
#from torchmetrics.audio.pesq import PESQ
from model.pit_wrapper import PITLossWrapper
from torch import nn
from model.combined_loss import BinaryCrossEntropyLoss_Mean
from model.stoi_metric import stoi
import numpy as np
from joblib import Parallel, delayed
import joblib 

def reorder_source_vad(preds, batch_indices):
        r"""Reorder sources according to the best permutation.

        Args:
            preds (torch.Tensor): Tensor of shape :math:`[B, num_spk, num_frames]`
            batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
                Contains optimal permutation indices for each batch.

        Returns:
            :class:`torch.Tensor`: Reordered sources.
        """
        reordered_sources = torch.stack(
            [torch.index_select(s, 0, b) for s, b in zip(preds, batch_indices)]
        )
        return reordered_sources


def reorder_source_vad_fast(preds, batch_indices):
    r"""Reorder sources according to the best permutation.

    Args:
        preds (torch.Tensor): Tensor of shape :math:`[B, num_spk, num_frames]`
        batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
            Contains optimal permutation indices for each batch.

    Returns:
        :class:`torch.Tensor`: Reordered sources.
    """
    batch_indices_r = batch_indices.repeat((preds.shape[-1],1,1))
    batch_indices_r_p = batch_indices_r.permute((1,2,0))
    reordered_sources = torch.gather(preds, dim=1, index=batch_indices_r_p)
    return reordered_sources


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


def signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
    r"""Signal-to-noise ratio (SNR_):
    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}
    where  :math:`P` denotes the power of each signal. The SNR metric compares the level
    of the desired signal to the level of background noise. Therefore, a high value of
    SNR means that the audio is clear.
    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        zero_mean:
            if to zero mean target and preds or not
    Returns:
        snr value of shape [...]
    Example:
        #>>> from torchmetrics.functional.audio import signal_noise_ratio
        #>>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        #>>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        #>>> signal_noise_ratio(preds, target)
        tensor(16.1805)
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

    noise = target - preds

    snr_value = (torch.sum(target ** 2, dim=-1) + EPS) / (torch.sum(noise ** 2, dim=-1) + EPS)
    snr_value = 10 * torch.log10(snr_value)

    return snr_value

class SI_SDRi(nn.Module):
    def __init__(self):
        super().__init__()
        self.si_sdr_func = scale_invariant_signal_distortion_ratio
        self.pit_si_sdr_func = pit_si_sdr
    def forward(self, preds, target, mix):
        mix = mix.unsqueeze(dim=1)
        mix = mix.repeat(1, 2, 1)
        si_sdr_spks_start = self.si_sdr_func(mix, target, zero_mean=True)
        si_sdr_mix_start = torch.mean(si_sdr_spks_start)
        si_sdr = self.pit_si_sdr_func(preds, target)
        #if return2:
            #return si_sdr - si_sdr_mix_start, si_sdr_mix_start
        #else:
        #    return si_sdr - si_sdr_mix_start
        return si_sdr - si_sdr_mix_start


class Accuracy_Vad(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, targets, batch_indices_vad):
        """
        preds: with shape=[B, num_spk * 1,  T]
        targets: with shape=[B, num_spk * 1, T]
        """
        #preds = reorder_source_vad_fast(preds ,batch_indices_vad)
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        acc = torch.sum(preds == targets) / torch.numel(targets)
        acc0 = torch.sum(preds[:, 0] == targets[:, 0]) / torch.numel(targets[:, 0])
        acc1 = torch.sum(preds[:, 1] == targets[:, 1]) / torch.numel(targets[:, 1])
        return acc, acc0, acc1
    
    
class Accuracy_VadSum(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, targets):
        """
        preds: with shape=[B, T]
        targets: with shape=[B, T]
        """
        targets[targets == 2] = 1
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        acc = torch.sum(preds == targets) / torch.numel(targets)
        return acc


class Accuracy_Csd(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, targets):
        """
        preds: with shape=[B, num_class, T]
        targets: with shape=[B, T]
        """
        preds_class = torch.argmax(preds, dim=1)
        acc = torch.sum(preds_class == targets) / torch.numel(targets)
        return acc

class Stoi(nn.Module):
    def __init__(self):
        super().__init__()
        self.stoi = stoi
    def forward(self, preds, targets, batch_indices_separation):
        """
        preds: with shape=[B, num_spk * 1,  T]
        targets: with shape=[B, num_spk * 1, T]
        """
        
        B, num_spk, num_samples = preds.shape
        d = np.zeros((B, num_spk))
        for b in range(B):
            for s in range(num_spk):
                d[b, s] = stoi(targets[b, s].detach().cpu().numpy(), preds[b, s].detach().cpu().numpy(), 16000)
        mean_d = np.mean(d)   
        return mean_d

"""
class Stoi(nn.Module):
    def __init__(self):
        super().__init__()
        self.stoi = stoi
    def forward(self, preds, targets, batch_indices_separation):
        ""
        preds: with shape=[B, num_spk * 1,  T]
        targets: with shape=[B, num_spk * 1, T]
        ""
        number_of_cpu = joblib.cpu_count()
        delayed_funcs1 = [delayed(initial_sdr_whamr)(i, path) for i, path in enumerate(pickle_pathes)]
        torch.tensor(parallel_pool(delayed_funcs1)).numpy()
        preds = reorder_source_vad(preds, batch_indices_separation)
        B, num_spk, num_samples = preds.shape
        d = np.zeros((B, num_spk))
        for b in range(B):
            for s in range(num_spk):
                d[b, s] = stoi(targets[b, s].detach().cpu().numpy(), preds[b, s].detach().cpu().numpy(), 16000)
        mean_d = np.mean(d)   
        return mean_d"""


"""def si_sdri(preds, target, mix):
    mix = mix.unsqueeze(dim=1)
    mix = mix.repeat(1, 2, 1)
    si_sdr_mix = torch.mean(scale_invariant_signal_distortion_ratio(preds, mix))
    si_sdr = pit_si_sdr(preds, target)
    return si_sdr - si_sdr_mix"""

pit_snr = PIT(signal_noise_ratio, 'max', zero_mean=True)
pit_snr.__name__ = 'pit_snr'

pit_si_sdr = PIT(scale_invariant_signal_distortion_ratio, 'max', zero_mean=True) #zero_mean is True by defualt in pairwise_neg_sisdr
pit_si_sdr.__name__ = 'pit_si_sdr'

si_sdri = SI_SDRi()
si_sdri.__name__ = "si_sdri"

pit_CE_vad = PITLossWrapper(loss_func=BinaryCrossEntropyLoss_Mean(), pit_from='pw_pt')
pit_CE_vad.__name__ = 'pit_CE_vad'

csd_accuracy = Accuracy_Csd()
csd_accuracy.__name__ = "csd_accuracy"

vad_accuracy = Accuracy_Vad()
vad_accuracy.__name__ = "vad_accuracy"

vad_sum_accuracy = Accuracy_VadSum()
vad_sum_accuracy.__name__ = "vad_sum_accuracy"

stoi_met = Stoi()
stoi_met.__name__ = "stoi_metric"

stoi

if __name__ == '__main__':

    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    targets = torch.randint(0, 100, (1,2,48000)).float()
    st_targets = torch.randint(0, 100, (1,2,48000)).float()
    print(pit_si_sdr(targets, st_targets))
    print(loss_func(targets, st_targets))
    print("")

    targets = torch.randint(0, 100, (1, 2, 48000)).float()
    st_targets = torch.randint(0, 100, (1, 2, 48000)).float()
    loss_func = pairwise_neg_sisdr
    met = scale_invariant_signal_distortion_ratio(targets, st_targets, zero_mean=True)
    print(met)
    print(loss_func(targets, st_targets))
    