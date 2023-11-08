from torch.utils.data import Dataset
import pandas as pd
import torch
import pickle
import numpy as np
from scipy.io.wavfile import read
import os 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate


class Realistic_dataset(Dataset):
    def __init__(self, csv_file, csd_labels_freq, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csd_labels_freq = csd_labels_freq
        self.recording_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        a = 100
        return a
        #return self.recording_df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print('idx:', idx)

        record_path = self.recording_df.loc[idx, "path_file"]
        with open(record_path, "rb") as f:
            mix_without_noise, noisy_signal, _, speakers_target, s_thetas_array = pickle.load(f) 
            
        scenario = self.recording_df.loc[idx, "scenario"]
        reverb = self.recording_df.loc[idx, "rt60"]
        snr = self.recording_df.loc[idx, "mic_snr"]#it's actually wham snr
        path_label_freq_idx = os.path.join(self.csd_labels_freq, "scenario_{0}.npz".format(scenario))
        label = np.load(path_label_freq_idx)
        
        sample_separation = {'mix_without_noise': mix_without_noise[0], \
                             'mixed_signals': noisy_signal, 'clean_speeches': speakers_target,\
                                  "doa": s_thetas_array, "reverb":reverb, "snr":snr}
        label_csd = {"vad_frames_sum": label["vad_frames_sum"],\
                      "vad_frames_individual": label["vad_frames_individual"]}
        return sample_separation, label_csd



class Whamr_dataset(Dataset):
    def __init__(self, path_mix, csd_labels_freq="", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.recording_df = pd.read_csv(path_mix)
        self.transform = transform

    def __len__(self):
        # a = 100
        # return a
        return self.recording_df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        reverb_signals_targets1_path = self.recording_df.loc[idx, "path_file_spk1"]
        reverb_signals_targets2_path = self.recording_df.loc[idx, "path_file_spk2"]
        mixed_sig_np_path = self.recording_df.loc[idx, "path_file"]
        _, reverb_signals_targets1 = read(reverb_signals_targets1_path)
        _, reverb_signals_targets2 = read(reverb_signals_targets2_path)
        reverb_signals_targets = np.vstack((reverb_signals_targets1, reverb_signals_targets2)).copy()
    
        _, mixed_sig_np = read(mixed_sig_np_path)
        mixed_sig_np = mixed_sig_np.copy()

        reverb = self.recording_df.loc[idx, "rt60"]
        snr = self.recording_df.loc[idx, "mic_snr"]  # it's actually wham snr

        sample_separation = {'mix_without_noise': 0, 'mixed_signals': mixed_sig_np, 'clean_speeches': reverb_signals_targets,  "doa": 0,
                            "reverb": reverb, "snr": snr}
        label_csd = {"vad_frames_sum": 0, "vad_frames_individual": 0}

        return sample_separation, label_csd



if __name__ == '__main__':
    csd_labels_freq = "/dsi/gannot-lab/datasets/mordehay/data_wham_libri_all_overlap_shlomi/train/labels_npz/"
    data = Realistic_dataset("/mnt/dsi_vol1/shared/sharon_db/mordehay/train/csv_files/with_white_noise_res.csv", csd_labels_freq)
   
    indx = np.arange(4)
    datloader = DataLoader(data, batch_size=2, shuffle=False, sampler=SubsetRandomSampler(indx), collate_fn=default_collate)
    for sample in data:
        print("bb")
    print("dddd")
