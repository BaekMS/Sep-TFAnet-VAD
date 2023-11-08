
import torch
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from model.combined_loss import reorder_source_mse


class OnlineSaving:
    def __init__(self, model, save_path, criterion_similarity=None) -> None:
        self.indx = 0
        self.fs = 16000
        self.max_len = 3
        self.save_sec = 1
        self.model = model
        self.save_path = save_path
        self.online_sisdr = []
        self.reference_sisdr = []
        self.num_save_samples = 30
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
            
    def get_truncated_signal(self, full_signal_mix):
        truncated_signal_mix = full_signal_mix[:, int(np.floor(self.fs * self.indx * self.save_sec)): int(np.floor(self.fs * self.indx * self.save_sec)) + self.max_len * self.fs]
        return truncated_signal_mix
            
    def increase_indx(self):
        self.indx += 1
        
    def get_indx(self):
        return self.indx
    
    def save_audio(self, name_folder, separated_signals, mix):
        
        separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
        separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
        mix_waves = mix[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
        Path(f"{self.save_path}/{name_folder}/indx_{self.indx}").mkdir(parents=True, exist_ok=True)
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}/mixed.wav", self.fs, mix_waves.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}/output_0.wav", self.fs, separated_audio1.astype(np.float32))
        write(f"{self.save_path}/{name_folder}/indx_{self.indx}/output_1.wav", self.fs, separated_audio2.astype(np.float32))
        
        
    def save_last_online_audio(self, name_folder, online_signal, mixed_signal_t):
        
        online_signal = online_signal[0, :, :].cpu().detach().numpy() #sample 0 from batch
        mixed_signal_t = mixed_signal_t[0, :].cpu().detach().numpy() #sample 0 from batch
        Path(f"{self.save_path}/{name_folder}").mkdir(parents=True, exist_ok=True)
        #save online signal
        write(f"{self.save_path}/{name_folder}/online_signal0.wav", self.fs, online_signal[0].astype(np.float32))
        write(f"{self.save_path}/{name_folder}/online_signal1.wav", self.fs, online_signal[1].astype(np.float32))
        #save true signal
        
        write(f"{self.save_path}/{name_folder}/ref_mix.wav", self.fs, mixed_signal_t.astype(np.float32))
        
    def calc_online(self, full_signal_mix, name_folder, sample_indx, inference_kw):
        if full_signal_mix.shape[-1] < self.fs * self.max_len:
            full_signal_mix = torch.nn.functional.pad(full_signal_mix, (0, self.fs * self.max_len - full_signal_mix.shape[-1]))
            
        #print(full_signal_mix.shape[-1])
        max_indx = np.floor(((full_signal_mix.shape[-1] - self.fs * self.max_len) / (self.fs * self.save_sec)))
        #print(max_indx)
        
        while self.indx <= max_indx:

            truncated_signal_mix = self.get_truncated_signal(full_signal_mix)
            with torch.no_grad():
                pred_separation, _, _ = self.model(truncated_signal_mix, inference_kw)
            if self.indx == 0:
                self.update_online_signal(pred_separation)
            pred_separation_sim = pred_separation[:, :, - int(np.floor(self.fs*self.save_sec)) - self.online_signal.shape[-1]: - int(np.floor(self.fs*self.save_sec))]
            truncated_onlinet_sim = self.online_signal[:, :, - self.fs * self.max_len + int(np.floor(self.fs*self.save_sec)): ]

            _, batch_indices_separation = self.criterion_similarity(pred_separation_sim, truncated_onlinet_sim,
                                                                            return_incides=True)
            
            pred_separation = reorder_source_mse(pred_separation, batch_indices_separation)
            self.update_online_signal(pred_separation)
            if sample_indx < self.num_save_samples:
                self.save_audio(name_folder, pred_separation, truncated_signal_mix)
            self.increase_indx()

       
        mixed_signal_t = full_signal_mix[:, int(np.floor(self.fs * (self.max_len - self.save_sec))): int(np.floor(self.fs * (self.max_len + (self.indx - 1) * self.save_sec)))]
        #mixed_signal_t = mixed_signal_t[:, 6*self.fs:] ##test
        if sample_indx < self.num_save_samples:
            self.save_last_online_audio(name_folder, self.online_signal, mixed_signal_t)
        
        self.reset()

