import torch
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from model.combined_loss import reorder_source_mse
from model.combined_loss import calc_sisdr


class OnlineSaving:
    def __init__(self, criterion_separation, model, save_path, device, criterion_similarity=None) -> None:
        self.indx = 0
        self.fs = 16000
        self.max_len = 3
        self.save_sec = 1
        self.criterion_separation = criterion_separation
        self.model = model
        self.save_path = save_path
        self.online_sisdr = []
        self.reference_sisdr = []
        self.num_save_samples = 2000000
        self.similarity = False
        if criterion_similarity is not None:
            self.similarity = True
            self.criterion_similarity = criterion_similarity
        
            
    def reset(self):
        self.indx = 0
    def update_online_signal(self, est_signals):
        """_summary_

        Args:
            signals: the signal are after ordering (PIT)
        """
        if self.indx == 0:
            self.online_signal = est_signals[:, :, est_signals.shape[-1] - int(np.floor(self.fs*self.save_sec)):]
        else:
            self.online_signal = torch.cat((self.online_signal, est_signals[:, :, est_signals.shape[-1] - int(np.floor(self.fs * self.save_sec)):]), dim=-1)
            
    def get_truncated_signal(self, full_signal_mix, target_signal):
        truncated_signal_mix = full_signal_mix[:, int(np.floor(self.fs * self.indx * self.save_sec)): int(np.floor(self.fs * self.indx * self.save_sec)) + self.max_len * self.fs]
        truncated_signal_target = target_signal[:, :, int(np.floor(self.fs * self.indx * self.save_sec)): int(np.floor(self.fs * self.indx * self.save_sec)) + self.max_len * self.fs]        
        return truncated_signal_mix, truncated_signal_target
            
    def increase_indx(self):
        self.indx += 1
        
    def get_indx(self):
        return self.indx
    
    def save_audio(self, name_folder, separated_signals, target, mix, si_sdr):
        target_audio1 = target[0, 0, :].cpu().detach().numpy() #sample 0 from batch
        target_audio2 = target[0, 1, :].cpu().detach().numpy() #sample 0 from batch
        separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
        separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
        mix_waves = mix[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
        Path(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}").mkdir(parents=True, exist_ok=True)
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/mixed.wav", self.fs, mix_waves.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/output_0.wav", self.fs, separated_audio1.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/output_1.wav", self.fs, separated_audio2.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/clean_0.wav", self.fs, target_audio1.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}_sisdr_{si_sdr:.2f}/clean_1.wav", self.fs, target_audio2.astype(np.float32))
        
        
    def save_last_online_audio(self, name_folder, online_signal, true_truncated_signal, online_sisdr, ref_sisdr, mixed_signal_t):
        true_truncated_signal = true_truncated_signal[0, :, :].cpu().detach().numpy() #sample 0 from batch
        online_signal = online_signal[0, :, :].cpu().detach().numpy() #sample 0 from batch
        mixed_signal_t = mixed_signal_t[0, :].cpu().detach().numpy() #sample 0 from batch
        Path(f"{self.save_path}/{name_folder}").mkdir(parents=True, exist_ok=True)
        #save online signal
        asr_bool = True
        if asr_bool:
            f = open(f"{self.save_path}/{name_folder}/The Si-Sdr is {online_sisdr:.2f}.txt", "w")
            write(f"{self.save_path}/{name_folder}/online_signal0.wav", self.fs, online_signal[0].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/online_signal1.wav", self.fs, online_signal[1].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/ref_mix.wav", self.fs, mixed_signal_t.astype(np.float32))
        else:
            write(f"{self.save_path}/{name_folder}/online_signal0_onlineSisdr_{online_sisdr:.2f}_refSisdr_{ref_sisdr:.2f}.wav", self.fs, online_signal[0].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/online_signal1_onlineSisdr_{online_sisdr:.2f}_refSisdr_{ref_sisdr:.2f}.wav", self.fs, online_signal[1].astype(np.float32))
            #save true signal
            write(f"{self.save_path}/{name_folder}/true_signal0_onlineSisdr_{online_sisdr:.2f}_refSisdr_{ref_sisdr:.2f}.wav", self.fs, true_truncated_signal[0].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/true_signal1_onlineSisdr_{online_sisdr:.2f}_refSisdr_{ref_sisdr:.2f}.wav", self.fs, true_truncated_signal[1].astype(np.float32))
            write(f"{self.save_path}/{name_folder}/ref_mix.wav", self.fs, mixed_signal_t.astype(np.float32))
        
    def calc_online(self, full_signal_mix, target_signal, name_folder, sample_indx):
        if full_signal_mix.shape[-1] < self.fs * self.max_len:
            full_signal_mix = torch.nn.functional.pad(full_signal_mix, (0, self.fs * self.max_len - full_signal_mix.shape[-1]))
            target_signal = torch.nn.functional.pad(target_signal, (0, self.fs * self.max_len - target_signal.shape[-1]))
        #print(full_signal_mix.shape[-1])
        #print(full_signal_mix.shape)
        #print(target_signal.shape)
        full_signal_mix = torch.nn.functional.pad(full_signal_mix, (int(self.fs * (self.max_len - self.save_sec)), 0))
        target_signal = torch.nn.functional.pad(target_signal, (int(self.fs * (self.max_len - self.save_sec)), 0))
        pad_zero = (full_signal_mix.shape[-1] - self.fs * self.max_len) % (self.fs * self.save_sec)
        if pad_zero:  
            full_signal_mix = torch.nn.functional.pad(full_signal_mix, (0, pad_zero))
            target_signal = torch.nn.functional.pad(target_signal, (0, pad_zero))
        max_indx = np.floor(((full_signal_mix.shape[-1] - self.fs * self.max_len) / (self.fs * self.save_sec)))
        #print(max_indx)
        #self.online_signal = target_signal[:, :, self.max_len * self.fs - int(np.floor(self.fs * 2 * self.save_sec)): self.max_len * self.fs - int(np.floor(self.fs * 1 * self.save_sec))] ######maybe i dont need this
        while self.indx <= max_indx:
            #print("INNNNN")
            truncated_signal_mix, truncated_signal_target = self.get_truncated_signal(full_signal_mix, target_signal)
            with torch.no_grad():
                pred_separation, _, _, _, _, _ = self.model(truncated_signal_mix)
            separation_loss, batch_indices_separation = self.criterion_separation(pred_separation, truncated_signal_target,
                                                                                return_incides=True)
            if self.similarity:
                if self.indx == 0:
                #self.online_signal = pred_separation[:, :, pred_separation.shape[-1] - int(np.floor(self.fs*self.save_sec)):]
                    self.update_online_signal(pred_separation)
                pred_separation_sim = pred_separation[:, :, - int(np.floor(self.fs*self.save_sec)) - self.online_signal.shape[-1]: - int(np.floor(self.fs*self.save_sec))]
                truncated_onlinet_sim = self.online_signal[:, :, - self.fs * self.max_len + int(np.floor(self.fs*self.save_sec)): ]
                #print(f"the pred is: {pred_separation_sim.shape}")
                #print(f"the online is: {truncated_onlinet_sim.shape}")
                _, batch_indices_separation = self.criterion_similarity(pred_separation_sim, truncated_onlinet_sim,
                                                                                return_incides=True)
            pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
            self.update_online_signal(pred_separation)
            if sample_indx < self.num_save_samples:
                pass ######
                #self.save_audio(name_folder, pred_separation, truncated_signal_target, truncated_signal_mix, -separation_loss)
            self.increase_indx()
        
        ##get refernce score
        pred_separation, _, _, _, _, _ = self.model(full_signal_mix)
        separation_loss, batch_indices_separation = self.criterion_separation(pred_separation, target_signal,
                                                                                return_incides=True)
        pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
        pred_separation_t = pred_separation[:, :, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        target_signal_t = target_signal[:, :, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        #pred_separation_t = pred_separation_t[:, :, 6*self.fs:] ##test
        #target_signal_t = target_signal_t[:, :, 6*self.fs:] ##test
        ref_pred_and_true_signal_sisdr = torch.mean(calc_sisdr(pred_separation_t, target_signal_t)).item()
        self.reference_sisdr.append(ref_pred_and_true_signal_sisdr)
        
        ##get online score
        true_truncated_signal = target_signal[:, :, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        #true_truncated_signal = true_truncated_signal[:, :, 6*self.fs:] ##test
        #self.online_signal = self.online_signal[:, :, 6*self.fs:] ##test
        _, batch_indices_separation = self.criterion_separation(self.online_signal, true_truncated_signal,
                                                                                return_incides=True)
        self.online_signal = reorder_source_mse(self.online_signal, batch_indices_separation)
        online_pred_and_true_signal_sisdr = torch.mean(calc_sisdr(self.online_signal, true_truncated_signal)).item()
        self.online_sisdr.append(online_pred_and_true_signal_sisdr)
        mixed_signal_t = full_signal_mix[:, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        #mixed_signal_t = mixed_signal_t[:, 6*self.fs:] ##test
        if sample_indx < self.num_save_samples:
            pass ###########
            #self.save_last_online_audio(name_folder, self.online_signal, true_truncated_signal, online_pred_and_true_signal_sisdr, ref_pred_and_true_signal_sisdr, mixed_signal_t)
        
        
        
        self.reset()
        