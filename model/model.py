import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from torchaudio import transforms
from collections import OrderedDict
import torch.nn.utils.weight_norm as wn
import math
#from  memory_profiler import profile
from typing import TypedDict

PI = torch.Tensor([math.pi])
    


class InputSpec(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512) -> None:
        super().__init__()
        self.spec= transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                      window_fn=torch.hann_window, power=None) 
        
    def forward(self, sig_time):
        stft = self.spec(sig_time)
        stft[:, 0, :] = 0 #remove DC
        return stft 
    
class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        """not being used in our implementation. used for causal case
        """
        super(cLN, self).__init__()
        self.eps = eps
        if trainable:

            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=False, causal=False, bool_drop=True,
                 drop_value=0.1, weight_norm=False):
        super(DepthConv1d, self).__init__()
        self.causal = causal
        self.skip = skip
        self.bool_drop = bool_drop
        if not weight_norm:
            self.conv1d = nn.Conv1d(input_channel, input_channel, 1)

            if self.causal:
                self.padding = (kernel - 1) * dilation
            else:
                self.padding = padding
            groups = int(hidden_channel/4) #int(hidden_channel/2) #for BN=256 #TODO
            self.dconv1d = nn.Conv1d(input_channel, hidden_channel, kernel, dilation=dilation,
                                    groups=groups,
                                    padding=self.padding) #Depth-wise convolution
            self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
            self.nonlinearity1 = nn.PReLU()
            self.nonlinearity2 = nn.PReLU()
            self.drop1 = nn.Dropout2d(drop_value)
            self.drop2 = nn.Dropout2d(drop_value)
            if self.causal:
                self.reg1 = cLN(hidden_channel, eps=1e-08)
                self.reg2 = cLN(hidden_channel, eps=1e-08)
            else:
                self.reg1 = nn.GroupNorm(1, input_channel, eps=1e-08)
                self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

            if self.skip:
                self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

        else:
            self.conv1d = wn(nn.Conv1d(input_channel, input_channel, 1))

            if self.causal:
                self.padding = (kernel - 1) * dilation
            else:
                self.padding = padding
            groups = input_channel
            self.dconv1d = wn(nn.Conv1d(input_channel, hidden_channel, kernel, dilation=dilation,
                                    groups=groups,
                                    padding=self.padding)) #Depth-wise convolution
            self.res_out = wn(nn.Conv1d(hidden_channel, input_channel, 1))
            self.nonlinearity1 = nn.PReLU()
            self.nonlinearity2 = nn.PReLU()
            self.drop1 = nn.Dropout2d(drop_value)
            self.drop2 = nn.Dropout2d(drop_value)
            if self.causal:
                self.reg1 = cLN(input_channel, eps=1e-08)
                self.reg2 = cLN(hidden_channel, eps=1e-08)
            else:
                self.reg1 = nn.GroupNorm(1, input_channel, eps=1e-08)
                self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

            if self.skip:
                self.skip_out = wn(nn.Conv1d(hidden_channel, input_channel, 1))

    #@profile
    def forward(self, input):
        if self.bool_drop:
            output = self.reg1(self.drop1(self.nonlinearity1(self.conv1d(input))))#with dropout
            if self.causal:
                output = self.reg2(self.drop2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding])))
            else:
                output = self.reg2(self.drop2(self.nonlinearity2(self.dconv1d(output))))
        else:
            output = self.reg1(self.nonlinearity1(self.conv1d(input)))#without droput
            if self.causal:
                output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
            else:
                output = self.reg2(self.nonlinearity2(self.dconv1d(output))) #without dropout 
        
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual #In our implementation we don't use skip connection
            


class VAD(nn.Module):
    def __init__(self, fft_num=257, num_spk=2, weight_norm=True):
        super().__init__()
        self.num_spk = num_spk
        if weight_norm:
            self.common = nn.Sequential(OrderedDict([
            ('conv1_1', wn(nn.Conv1d(in_channels=fft_num, out_channels=4, kernel_size=5, stride=1, padding=5//2))),
            ('relu_1', nn.PReLU()),
            ('BN_1', nn.GroupNorm(1, 4, eps=1e-8))
            ]))
            self.output_layer_vad = wn(nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1))
        else:
            self.common = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(in_channels=fft_num, out_channels=4, kernel_size=5, stride=1, padding=5//2)),
            ('relu_1', nn.PReLU()),
            ('BN_1', nn.GroupNorm(1, 4, eps=1e-8))
            ]))
            self.output_layer_vad = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #input shape is [B, F*n_spk, num_frames]
        out_common = self.common(input)
        output_vad = self.output_layer_vad(out_common)#list of object with shape [B, 1, num_frames]
        output_vad = self.sigmoid(output_vad)

        return output_vad


class TF_Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1d_t_1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1d_t_2 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2, dilation=2)
        self.sigmoid_t = nn.Sigmoid()
        self.prelu_t = nn.PReLU()
        self.adapt_avrg_pooling_t = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1d_f_1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1d_f_2 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2, dilation=2)
        self.sigmoid_f = nn.Sigmoid()
        self.prelu_f = nn.PReLU()
        self.adapt_avrg_pooling_f = nn.AdaptiveAvgPool2d((None, 1))
        
    def forward(self, input):
        output_t = self.adapt_avrg_pooling_t(input) #[B, 1, T]
        output_t = self.sigmoid_t(self.prelu_t(self.conv1d_t_2(self.conv1d_t_1(output_t))))
        
        output_f = self.adapt_avrg_pooling_f(input) #[B, F, 1]
        output_f = torch.transpose(output_f, 1, 2) #[B, 1, F]
        output_f = self.sigmoid_f(self.prelu_f(self.conv1d_f_2(self.conv1d_f_1(output_f))))
        output_f = torch.transpose(output_f, 1, 2) #[B, F, 1]
        
        attention_w = output_f @ output_t #[B, F, T]
        output = input * attention_w 
        return output
 
class TCN(nn.Module): #this is the audio blocks
    def __init__(self, input_dim, output_dim, BN_dim, H_dim,
                layer, R, kernel=3, skip=False,
                causal=False, dilated=True, bool_drop=True, drop_value=0.1, weight_norm=False,
                tf_attention=False, apply_recursive_ln=False, apply_residual_ln=False):
        super(TCN, self).__init__()
        self.layer = layer
        self.tf_attention = tf_attention
        self.apply_recursive_ln = apply_recursive_ln
        self.apply_residual_ln = apply_residual_ln
        self.skip = skip

        # normalization
        if not weight_norm:
            if not causal:
                self.LN = nn.GroupNorm(1, input_dim, eps=1e-8) #this is like layer normalization because the number of groups is equal to one
            else:
                self.LN = cLN(input_dim, eps=1e-8)

            self.BN = nn.Conv1d(input_dim, BN_dim, 1)
            
            # TCN for feature extraction
            self.receptive_field = 0
            self.dilated = dilated

            self.TCN = nn.ModuleList([])
            if self.tf_attention:
                self.time_freq_attnetion = nn.ModuleList([])
            for r in range(R):
                for i in range(layer):
                    if self.dilated:
                        if i == 0:
                            self.TCN.append(DepthConv1d(BN_dim, H_dim, kernel, dilation=1, 
                                                        padding=1, skip=skip,
                                                        causal=causal, bool_drop=bool_drop, 
                                                        drop_value=drop_value, weight_norm=weight_norm)) 
                        else:
                            self.TCN.append(DepthConv1d(BN_dim, H_dim, kernel, dilation=2 * i,
                                                    padding=2 * i, skip=skip,
                                                    causal=causal, bool_drop=bool_drop, 
                                                    drop_value=drop_value, weight_norm=weight_norm)) ##I vhange to multiply and not square
                        if self.tf_attention:
                            self.time_freq_attnetion.append(TF_Attention())
                            
                    else:
                        self.TCN.append(
                            DepthConv1d(BN_dim, H_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal, bool_drop=bool_drop,
                                         drop_value=drop_value, weight_norm=weight_norm))
                    if i == 0 and r == 0:
                        self.receptive_field += kernel
                    else:
                        if self.dilated:
                            self.receptive_field += (kernel - 1) * (i % 4 + 1)
                        else:
                            self.receptive_field += (kernel - 1)

            # output layer
            self.output = nn.Sequential(nn.PReLU(),
                                    nn.GroupNorm(1, BN_dim),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                    )
        else:
            if not causal:
                #this is like layer normalization because the number of groups is equal to one input_dim
                self.LN = nn.GroupNorm(1, input_dim, eps=1e-8) 
            else:
                self.LN = cLN(input_dim, eps=1e-8)

            # TCN for feature extraction
            self.receptive_field = 0
            self.dilated = dilated

            self.TCN = nn.ModuleList([])
            if self.tf_attention:
                self.time_freq_attnetion = nn.ModuleList([])
            for r in range(R):
                for i in range(layer):
                    if self.dilated:
                        if i == 0:
                            self.TCN.append(DepthConv1d(BN_dim, H_dim, kernel, dilation=1, padding=1, skip=skip,
                                                        causal=causal, bool_drop=bool_drop, drop_value=drop_value, weight_norm=weight_norm)) 
                        else:
                            self.TCN.append(DepthConv1d(BN_dim, H_dim, kernel, dilation=i%4+1, padding=i%4+1, skip=skip,
                                                    causal=causal, bool_drop=bool_drop, drop_value=drop_value, weight_norm=weight_norm))
                        if self.tf_attention:
                            self.time_freq_attnetion.append(TF_Attention())

                        
                    else:
                        self.TCN.append(
                            DepthConv1d(BN_dim, H_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal, bool_drop=bool_drop,
                                        drop_value=drop_value, weight_norm=weight_norm))
                    if i == 0 and r == 0:
                        self.receptive_field += kernel
                    else:
                        if self.dilated:
                            self.receptive_field += (kernel - 1) * (i % 4 + 1)
                        else:
                            self.receptive_field += (kernel - 1)

            if self.apply_recursive_ln:
                self.ln_first_modules = nn.ModuleList([])
                self.ln_second_modules = nn.ModuleList([])
                for i in range(len(self.TCN)):
                    self.ln_first_modules.append(nn.GroupNorm(1, BN_dim))
                    self.ln_second_modules.append(nn.GroupNorm(1, BN_dim))
            if self.apply_residual_ln:
                self.ln_modules = nn.ModuleList([])
                for i in range(len(self.TCN)):
                    self.ln_modules.append(nn.GroupNorm(1, BN_dim))  
                    
            # output layer
            self.output = nn.Sequential(nn.PReLU(),
                                    nn.GroupNorm(1, BN_dim),
                                    wn(nn.Conv1d(BN_dim, output_dim, 1))
                                    )
            
                
    #@profile
    def forward(self, input):
        # input shape: (B, n_fft / 2, T)
        
        # normalization
        output = self.LN(input)
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:

            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                if self.tf_attention:
                    residual = self.time_freq_attnetion[i](residual)
                if self.apply_recursive_ln:
                    output = self.ln_second_modules[i](output + self.ln_first_modules[i](output + residual))
                elif self.apply_residual_ln:
                    output = output + self.ln_modules[i](residual)
                else:
                    output = output + residual
        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        return output

class SeparationModel(nn.Module):
    def __init__(self, **config):
        defaults  = {'n_fftBins': 512, 'BN_dim': 256, 'H_dim': 512, 'layer': 8, 'stack': 3, 'kernel': 3,
                            'num_spk': 2, 'skip': False, 'dilated': True, 'casual': False, 'bool_drop': True,
                            'drop_value': 0.1, 'weight_norm': False, 'final_vad': True, 'noisy_phase': False,
                            'activity_input_bool': False, 'tf_attention': False, 'apply_recursive_ln': False,
                            'apply_residual_ln': False, 'final_vad_masked_speakers': False}

        super(SeparationModel, self).__init__()
        
        defaults.update(config)
        print(defaults)
        # Set attributes from the dictionary.
        for key, value in defaults.items():
            setattr(self, key, value)

        
        self.n_fftBins_h = self.n_fftBins // 2 + 1
        hop_length = int(self.n_fftBins/2) #32 self.
        input_tcn = self.n_fftBins_h - 1 #the one is the DC
        output_tcn = self.n_fftBins_h * self.num_spk
        
        self.am_to_db = transforms.AmplitudeToDB(stype="power")
        self.spec_input = InputSpec(n_fft=self.n_fftBins, hop_length=hop_length, win_length=self.n_fftBins)
        self.spec_output = transforms.Spectrogram(n_fft=self.n_fftBins, hop_length=hop_length, win_length=self.n_fftBins,
                                      window_fn=torch.hann_window, power=None)
        self.inv_spec = transforms.InverseSpectrogram(n_fft=self.n_fftBins, hop_length=hop_length, win_length=self.n_fftBins,
                                    window_fn=torch.hann_window)
        self.TCN = TCN(input_tcn, output_tcn, self.BN_dim, self.H_dim,
                    self.layer, self.stack, kernel=3, skip=self.skip,
                    causal=self.casual, dilated=self.dilated, bool_drop=self.bool_drop, drop_value=self.drop_value, weight_norm=self.weight_norm,
                    tf_attention=self.tf_attention, apply_recursive_ln=self.apply_recursive_ln, apply_residual_ln=self.apply_residual_ln)
                    
        #self.to_spec = nn.Conv1d(self.BN_dim, self.num_spk*self.n_fftBins_h, 1)
        self.m = nn.Sigmoid()
        if self.final_vad:
            self.vad = VAD(fft_num=int(self.n_fftBins/2) + 1)
        
        if self.activity_input_bool:
            self.activity_input = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=1, padding=(1,1))
            self.prelu = nn.PReLU()  
    
    def forward(self, x: torch.float32, inference_kw={}):
        """
        x - 2-speaker mixed signal, shape = [B, T]
        """
        assert x.ndim == 2 , "input tensor must be 2 dimensions (B, T), but got dimensions of {}".format(x.ndim)            
        num_of_samples = x.shape[-1]
        stft_out = self.spec_output(x)
        stft = self.spec_input(x)
        stft_out[:, 0, :] = 0 #remove DC. In Librispeech dataset the DC is not zero
        power = torch.pow(torch.abs(stft), 2)
        self.spectrum = self.am_to_db(power) #shape = [B, 257, T]
        ####Adding  activity in input
        if self.activity_input_bool:
            y = torch.unsqueeze(self.spectrum, dim=1)
            x = self.activity_input(y)
            spec_activity = torch.squeeze(x, dim=1)
            spec_activity = self.prelu(spec_activity)
            self.spectrum = self.spectrum * spec_activity
        
        self.masks_b = self.TCN(self.spectrum[:, 1:]) #All frequncies except DC. [B, , self.n_fftBins_h * self.num_spk, T]. In this stage the masks are not sigmoided
        batch_size, _, frames_number = self.masks_b.shape 
        self.mask_per_speaker = self.masks_b.reshape(batch_size, self.num_spk, self.n_fftBins_h, frames_number)
        if self.final_vad and not self.final_vad_masked_speakers: #The VAD is calculated on the masks
            output_vad = torch.cat([self.vad(self.mask_per_speaker[:, s]) for s in range(self.num_spk)], dim=1) # shape = [B, self.num_spk, T]
        else:
            output_vad = 0

        self.mask_per_speaker = self.m(self.mask_per_speaker)
        if self.noisy_phase:
            stft_mags = stft_out.abs()
            noisy_stft_phase = stft_out.angle()
            masked_mag = torch.mul(torch.unsqueeze(stft_mags, 1), self.mask_per_speaker) #[B, 2, 257, T]
            if self.final_vad and self.final_vad_masked_speakers: #The VAD is calculated on the masked spectrum
                output_vad = torch.cat([self.vad(masked_mag[:, s]) for s in range(self.num_spk)], 
                                       dim=1) #self.vad(self.masks_b) need to change for regular vad
            self.estimated_stfts = masked_mag.type(torch.complex64) * torch.unsqueeze(torch.exp(1j*noisy_stft_phase), 1) # Attaching the noisy phase
        else:
            self.estimated_stfts = torch.mul(torch.unsqueeze(stft_out, 1), self.mask_per_speaker) #shape = [B, 2, F, T]

        
        ########################### Only in inference

        if inference_kw and self.final_vad:
            filter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=inference_kw["length_smoothing_filter"], stride=1, padding=1, bias=False)
            w_x = torch.tensor([[1., 0., 1.]])
            w_x = w_x.unsqueeze(0)
            filter.weight = nn.Parameter(w_x, requires_grad=False)
            threshold_vad = torch.where(torch.unsqueeze(output_vad, dim=2)>=inference_kw["threshold_activated_vad"], 1., 0.)
            vad_smooth = torch.minimum(torch.stack([filter(threshold_vad[:, s]) for s in range(self.num_spk)], dim=1), torch.tensor(1, device=threshold_vad.device))
            vad_smooth[..., [0, -1]] = threshold_vad[..., [0, -1]]
            if inference_kw["filter_signals_by_smo_vad"]:  
                self.estimated_stfts = vad_smooth * self.estimated_stfts
            elif inference_kw["filter_signals_by_unsmo_vad"]:
                self.estimated_stfts = vad_smooth * self.estimated_stfts
            if inference_kw["return_smoothed_vad"]:
                output_vad = vad_smooth
        ########################### Only in inference
        
        out_separation = self.inv_spec(self.estimated_stfts, length=num_of_samples)  # shape = [B, self.num_spk, samples]                            
        return out_separation, output_vad, self.estimated_stfts
    

if __name__ == '__main__':
    s = SeparationModel()
    x = torch.rand(size=(10, 6, 48000))
    out = s(x)
    print("done")
    