from turtle import forward
from torch import nn
import torch
from torchaudio import transforms
from torch import Tensor
import numpy as np
import torch.nn.functional as F
from functools import partial


def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(f"Predictions and targets are expected to have the same shape, pred has shape of {preds.shape} and target has shape of {target.shape}")

def calc_sisdr(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
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
        torch.sum(target ** 2, dim=-1, keepdim=True) + EPS
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled ** 2, dim=-1) + EPS) / (torch.sum(noise ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)

    return val

def calc_sisdr_loss(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
    val = calc_sisdr(preds, target, zero_mean)
    return -val


def reorder_source_mse(preds, batch_indices):
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

class CombinedLoss(nn.Module):
    def __init__(self, criterion_separation, criterion_vad, weights):
        super().__init__()
        self.criterion_separation = criterion_separation
        self.criterion_vad = criterion_vad
        self.bce = BinaryCrossEntropyLoss_Mean()
        self.weights = weights
        self.learn_weight_bool = weights["learn_weight_bool"]
        if self.learn_weight_bool: #for learning the weights between vad loss to separation loss
            self.learn_weight_vadLoss = nn.parameter.Parameter(torch.tensor(0.5))
            self.learn_weight_separationLoss = nn.parameter.Parameter(torch.tensor(0.5))
            

    def forward(self, pred_separation, target_separation, pred_vad, target_vad):
 
        separation_loss, batch_indices_separation = self.criterion_separation(pred_separation, target_separation,
                                                                            return_incides=True) #batch_indices_separation is the \
                                                                                #permutation matrix with shape [B, num_spk].\
                                                                                    # This matrix can be applied on the predicted signals in order to\
                                                                                        # be aligned with the target signals. 
        pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
        if self.learn_weight_bool:
            separation_loss = separation_loss * (1 / (2 * torch.square(self.learn_weight_separationLoss))) + \
                torch.log(1 + torch.square(self.learn_weight_separationLoss))


        if self.weights["weight_vad_loss"]:
            #print(pred_vad.shape)
            #print(target_vad.shape)
            pred_vad = reorder_source_mse(pred_vad, batch_indices_separation)
            vad_loss = self.bce(pred_vad, target_vad)
            if self.learn_weight_bool:
                vad_loss = vad_loss * (1 / (2 * torch.square(self.learn_weight_vadLoss))) + torch.log(1 + torch.square(self.learn_weight_vadLoss))
            #vad_loss, batch_indices_vad = self.criterion_vad(pred_vad, target_vad, return_incides=True)
            batch_indices_vad = batch_indices_separation
        else:
            vad_loss, batch_indices_vad = torch.tensor(0), torch.tensor(0)

        return separation_loss, vad_loss, batch_indices_vad, batch_indices_separation
        
class BinaryCrossEntropyLoss_Mean(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCE_none = nn.BCELoss(reduction='none')
    def forward(self, pred_vad, target_vad):
        """
        pred_csd.shape = [B, num_frames]
        target_csd.shape = [B, num_frames]
        """
    
        target_vad = target_vad.to(torch.float32) 
        target_vad[target_vad == 2] = 1
        out = self.BCE_none(pred_vad, target_vad) #shape = [B, num_frames]
        ### For weighting
        weight = target_vad * 0.36 + 0.64 * (1 - target_vad) #Here we give more weight to the 0's since they are less frequent 
        out = out * weight
        ###
        output = torch.mean(torch.sum(out, dim=(-1, -2)).to(torch.float32)) #mean over the batch, sum over the frames and speakers
        return output