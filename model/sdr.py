import torch
from torch.nn.modules.loss import _Loss
from torchaudio import transforms



class PairwiseNegSDR(_Loss):
    r"""Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target
            and estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.

    Shape:
        - est_targets : :math:`(batch, nsrc, ...)`.
        - targets: :math:`(batch, nsrc, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.

    Examples

        import torch
        from asteroid.losses import PITLossWrapper
        targets = torch.randn(10, 2, 32000)
        st_targets = torch.randn(10, 2, 32000)
        loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                                    pit_from='pairwise')
        loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super(PairwiseNegSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True) #[B, n_src, 1]
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source #[B, n_src, samples]
            est_targets = est_targets - mean_estimate #[B, n_src, samples]
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1) #[B, 1, n_src, samples]
        s_estimate = torch.unsqueeze(est_targets, dim=2) #[B, n_src, 1, samples]

        if self.sdr_type in ["sisdr", "sdsdr"]:
            
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True) #[batch, n_src, n_src, 1]
            
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS #[batch, 1, n_src, 1]
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
            #pair_wise_sdr = torch.clamp(pair_wise_sdr, max=9) ###Addddddddddddddd
            #pair_wise_sdr = torch.where(pair_wise_sdr >=0, -torch.clamp(pair_wise_sdr, max=8), pair_wise_sdr**2)
        return -pair_wise_sdr
        #return pair_wise_sdr


class SingleSrcNegSDR(_Loss):
    r"""Base class for single-source negative SI-SDR, SD-SDR and SNR.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target and
            estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.
        reduction (string, optional): Specifies the reduction to apply to
            the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

    Shape:
        - est_targets : :math:`(batch, time)`.
        - targets: :math:`(batch, time)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)` if ``reduction='none'`` else
        [] scalar if ``reduction='mean'``.

    Examples

        import torch
        from asteroid.losses import PITLossWrapper
        targets = torch.randn(10, 2, 32000)
        est_targets = torch.randn(10, 2, 32000)
        oss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
                                    pit_from='pw_pt')
        oss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, reduction="none", EPS=1e-8):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, est_target, target):
        if target.size() != est_target.size() or target.ndim != 2:
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(est_target * target, dim=1, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + self.EPS
            # [batch, time]
            scaled_target = dot * target / s_target_energy
        else:
            # [batch, time]
            scaled_target = target
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + self.EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses


class MultiSrcNegSDR(_Loss):
    r"""Base class for computing negative SI-SDR, SD-SDR and SNR for a given
    permutation of source and their estimates.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target
            and estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.

    Shape:
        - est_targets : :math:`(batch, nsrc, time)`.
        - targets: :math:`(batch, nsrc, time)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)` if ``reduction='none'`` else
        [] scalar if ``reduction='mean'``.

    Examples

        mport torch
        from asteroid.losses import PITLossWrapper
        targets = torch.randn(10, 2, 32000)
        est_targets = torch.randn(10, 2, 32000)
        loss_func = PITLossWrapper(MultiSrcNegSDR("sisdr"),
                                    pit_from='perm_avg')
        loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.

    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src]
            pair_wise_dot = torch.sum(est_targets * targets, dim=2, keepdim=True)
            # [batch, n_src]
            s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + self.EPS
            # [batch, n_src, time]
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            # [batch, n_src, time]
            scaled_targets = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_targets - targets
        else:
            e_noise = est_targets - scaled_targets
        # [batch, n_src]
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
            torch.sum(e_noise ** 2, dim=2) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -torch.mean(pair_wise_sdr, dim=-1)


class EmphasizedPairwiseNegSDR(_Loss):
    r"""Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target
            and estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.

    Shape:
        - est_targets : :math:`(batch, nsrc, ...)`.
        - targets: :math:`(batch, nsrc, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.

    Examples

        import torch
        from asteroid.losses import PITLossWrapper
        targets = torch.randn(10, 2, 32000)
        st_targets = torch.randn(10, 2, 32000)
        loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                                    pit_from='pairwise')
        loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8, n_fft=512):
        super(EmphasizedPairwiseNegSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS
        self.n_fft = n_fft
        self.spec = transforms.Spectrogram(n_fft=self.n_fft, hop_length=256, win_length=self.n_fft,
                                      window_fn=torch.hann_window, power=None)  # for all channels
        self.inv_spec = transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=256, win_length=self.n_fft,
                                      window_fn=torch.hann_window)

    def forward(self, est_targets, targets):
        """
        emphasized_mask with shape [B, n_src, F, T]
        """
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True) #[B, n_src, 1]
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source #[B, n_src, samples]
            est_targets = est_targets - mean_estimate #[B, n_src, samples]
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1) #[B, 1, n_src, samples]
        s_estimate = torch.unsqueeze(est_targets, dim=2) #[B, n_src, 1, samples]

        if self.sdr_type in ["sisdr", "sdsdr"]:
            
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True) #[batch, n_src, n_src, 1]
            
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS #[batch, 1, n_src, 1]

            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy #[batch, 1, n_src, samples]
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:

            ############
            #create emphasize mask for each speaker
            pitch = []
            target_stft = torch.transpose(self.spec(targets), dim0=-1, dim1=-2) #shape=[B, n_src, frames, F]
            target_stft[..., 0] = 0
            for i in range(target_stft.shape[1]):
                #target_stft[:,i,1] = 0
                pitch.append(stft_pitch_detect(target_stft[:,i]))
            emphasized_masks = torch.stack(pitch, dim=1) #shape=[B, n_src, frames, F]  
            emphasized_masks = torch.transpose(emphasized_masks, dim0=-1, dim1=-2) #shape=[B, n_src, F, frames] 
            #print(f"{torch.isnan(emphasized_masks).any()} in emphasized_masks")
            ############

            e_noise = s_estimate - pair_wise_proj #[B, n_src, 1, samples] - [batch, 1, n_src, samples] = [B, n_src, n_src, samples]
            num_of_samples = e_noise.shape[-1]
            e_noise_stft = self.spec(e_noise) #shape=[B, n_src, n_src, F, frames]
            emphasized_masks = torch.unsqueeze(emphasized_masks, dim=1) #[B, 1, n_src, F, T] (T is the frames)
            emphasized_masks[emphasized_masks < 0.4] = 0
            #print(f"{torch.isnan(e_noise_stft).any()} in e_noise_stft")
            #print(f"{torch.isnan(emphasized_masks).any()} in emphasized_masks")
            emphasized_e_noise = self.inv_spec(e_noise_stft * (1 + emphasized_masks), length=num_of_samples) #[B, n_src, n_src, samples]
        #print(f"{torch.isnan(pair_wise_proj).any()} in pair_wise_proj")
        #print(f"{torch.isnan(emphasized_e_noise).any()} in emphasized_e_noise")
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(emphasized_e_noise ** 2, dim=3) + self.EPS # [batch, n_src, n_src]
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -pair_wise_sdr




class Emphasized2PairwiseNegSDR(_Loss):
    r"""Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target
            and estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.

    Shape:
        - est_targets : :math:`(batch, nsrc, ...)`.
        - targets: :math:`(batch, nsrc, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.

    Examples

        import torch
        from asteroid.losses import PITLossWrapper
        targets = torch.randn(10, 2, 32000)
        st_targets = torch.randn(10, 2, 32000)
        loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                                    pit_from='pairwise')
        loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8, n_fft=512):
        super(Emphasized2PairwiseNegSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS
        self.n_fft = n_fft
        self.spec = transforms.Spectrogram(n_fft=self.n_fft, hop_length=256, win_length=self.n_fft,
                                      window_fn=torch.hann_window, power=None)  # for all channels
        self.inv_spec = transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=256, win_length=self.n_fft,
                                      window_fn=torch.hann_window)

    def forward(self, est_targets, targets):
        """
        emphasized_mask with shape [B, n_src, F, T]
        """
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True) #[B, n_src, 1]
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source #[B, n_src, samples]
            est_targets = est_targets - mean_estimate #[B, n_src, samples]
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)

            ############
            #create emphasize mask for each speaker
            pitch = []
            target_stft = torch.transpose(self.spec(targets), dim0=-1, dim1=-2) #shape=[B, n_src, frames, F]
            for i in range(target_stft.shape[1]):
                target_stft[:,i,1] = 0
                pitch.append(stft_pitch_detect(target_stft[:,i]))
            emphasized_masks = torch.stack(pitch, dim=1) #shape=[B, n_src, frames, F]  
            emphasized_masks = torch.transpose(emphasized_masks, dim0=-1, dim1=-2) #shape=[B, n_src, F, frames] 
            emphasized_masks[emphasized_masks < 0.4] = 0
            emphasized_masks[emphasized_masks >= 0.4] = 1
            num_of_samples = targets.shape[-1]
            targets = self.inv_spec(self.spec(targets) * (1 + emphasized_masks), length=num_of_samples)
            est_targets = self.inv_spec(self.spec(est_targets) * (1 + emphasized_masks), length=num_of_samples)
            #print(f"{torch.isnan(emphasized_masks).any()} in emphasized_masks")
            ############

        s_target = torch.unsqueeze(targets, dim=1) #[B, 1, n_src, samples]
        s_estimate = torch.unsqueeze(est_targets, dim=2) #[B, n_src, 1, samples]

        if self.sdr_type in ["sisdr", "sdsdr"]:
            
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True) #[batch, n_src, n_src, 1]
            
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS #[batch, 1, n_src, 1]



            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy #[batch, 1, n_src, samples]
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj #[B, n_src, 1, samples] - [batch, 1, n_src, samples] = [B, n_src, n_src, samples]
           
        #print(f"{torch.isnan(pair_wise_proj).any()} in pair_wise_proj")
        #print(f"{torch.isnan(emphasized_e_noise).any()} in emphasized_e_noise")
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS # [batch, n_src, n_src]
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -pair_wise_sdr


class PairwiseNegSDRClip(_Loss):
    r"""Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the target
            and estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.

    Shape:
        - est_targets : :math:`(batch, nsrc, ...)`.
        - targets: :math:`(batch, nsrc, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch, nsrc, nsrc)`. Pairwise losses.(the estimated is the rows)

    Examples

        import torch
        from asteroid.losses import PITLossWrapper
        targets = torch.randn(10, 2, 32000)
        st_targets = torch.randn(10, 2, 32000)
        loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
                                    pit_from='pairwise')
        loss = loss_func(est_targets, targets)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super(PairwiseNegSDRClip, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True) #[B, n_src, 1]
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source #[B, n_src, samples]
            est_targets = est_targets - mean_estimate #[B, n_src, samples]
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1) #[B, 1, n_src, samples]
        s_estimate = torch.unsqueeze(est_targets, dim=2) #[B, n_src, 1, samples]

        if self.sdr_type in ["sisdr", "sdsdr"]:
            
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True) #[batch, n_src, n_src, 1]
            
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS #[batch, 1, n_src, 1]
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
            
            #pair_wise_sdr = 8.3 * ((torch.exp(-pair_wise_sdr/8.3) - 1))
            #pair_wise_sdr = torch.maximum(torch.tensor(0., device=pair_wise_sdr.device), torch.exp(-0.4*pair_wise_sdr) - 1.) + torch.minimum(torch.tensor(0., device=pair_wise_sdr.device), 9. * (torch.exp(-pair_wise_sdr/9.) - 1.))
            #pair_wise_sdr = 12 * torch.tanh(pair_wise_sdr / 12)
            #pair_wise_sdr = 6.8*(torch.exp(-pair_wise_sdr/6.8) - 1)#I add
        return pair_wise_sdr #I add
        #return -pair_wise_sdr



##########new
# aliases
pairwise_neg_sisdr = PairwiseNegSDR("sisdr")
pairwise_neg_sisdr_clip = PairwiseNegSDRClip("sisdr")
pairwise_neg_emphasize_sisdr = EmphasizedPairwiseNegSDR("sisdr")
pairwise_neg_emphasize2_sisdr = Emphasized2PairwiseNegSDR("sisdr")
pairwise_neg_sdsdr = PairwiseNegSDR("sdsdr")
pairwise_neg_snr = PairwiseNegSDR("snr")
singlesrc_neg_sisdr = SingleSrcNegSDR("sisdr")
singlesrc_neg_sdsdr = SingleSrcNegSDR("sdsdr")
singlesrc_neg_snr = SingleSrcNegSDR("snr")
multisrc_neg_sisdr = MultiSrcNegSDR("sisdr")
multisrc_neg_sdsdr = MultiSrcNegSDR("sdsdr")
multisrc_neg_snr = MultiSrcNegSDR("snr")
