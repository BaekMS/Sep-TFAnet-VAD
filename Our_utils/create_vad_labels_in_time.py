import os
import torch
import numpy as np
import json
from tqdm import tqdm 
from pathlib import Path

#script for converting the VAD binary label from STFT to time domain
def load_config(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    return config_dict

# Directory paths
input_dir = "/dsi/gannot-lab1/datasets/mordehay/data_wham_partial_overlap/train/labels/"
output_dir = "/dsi/gannot-lab1/datasets/mordehay/data_wham_partial_overlap/train/labels_time/"
Path(output_dir).mkdir(parents=True, exist_ok=True)
length = 159744
# Iterate over files in the input directory
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.json'):  # Assuming the config file has a .json extension
        if filename[-1] == 1:
            continue
        save_a = np.zeros((2, length), dtype=np.float32)
        # Process the corresponding tensor
        filename_wo_ext = os.path.splitext(filename)[0]
        numpy_file = os.path.splitext(filename_wo_ext[:-6])[0] + '.npz'
        for spk in range(2): #for 2 speakers only
            # Load the config file
            config_file = os.path.join(input_dir, filename_wo_ext[:-1] + str(spk) + ".json")
            config_dict = load_config(config_file)
            speech_segments = config_dict['speech_segments']
            for i in range(len(speech_segments)):
                start_times = speech_segments[i]['start_time']
                end_times = speech_segments[i]['end_time']
                save_a[spk, start_times:end_times+1] = 1
        # Save the result in the output directory with the same file name
        output_file = os.path.join(output_dir, numpy_file)
        np.savez(output_file, time_activity=save_a)
