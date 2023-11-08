#Create lables from json files and saving as npz file 

import json
from pprint import pprint
import yaml
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm 
save_path = "/dsi/gannot-lab/datasets/mordehay/data_for_robot_one_speaker/test/labels_npz"  #need to change!!
Path(save_path).mkdir(parents=True, exist_ok=True)
save_each_spk = True

def convert_samples2frames_vad(samples, frame_len = 512):
    num_frames = int(2 * len(samples)/512 - 1)  #len(samples)/512 + (len(samples)/512 - 1)
    vad_frames = np.zeros(num_frames)
    size = 0
    for indx, step in enumerate(range(0, len(samples) - 256, int(frame_len/2))):#I assume hop length = 0.5 * frame_len
        temp = samples[step: step + frame_len]
        count_array = np.bincount(temp)
        vad_frames[indx] = np.argmax(count_array)
    
    return vad_frames


with open('/home/dsi/moradim/Audio-Visual-separation-using-RTF/create_data_frenkel/data_conifg_wham.yaml') as file: #need to change!!
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    args = yaml.load(file, Loader=yaml.FullLoader)

num_frames_without_hop = int(np.floor((args["aud_length"] * args["fs"]) / args["nfft"]))
cut_length = int(num_frames_without_hop * args["nfft"])
num_files = len(os.listdir("/dsi/gannot-lab/datasets/mordehay/data_for_robot_one_speaker/test/with_wham_noise_audio/")) #need to change!!
num_spk = 2
for num in tqdm(range(num_files)):
    vad = np.zeros((num_spk, cut_length), dtype=np.int64)
    num_frames = int(2 * cut_length/args["nfft"] - 1 + 2)  #len(samples)/512 + (len(samples)/512 - 1) + 2 (the 2 frames are the edges)
    vad_frames = np.zeros((num_spk, num_frames))
    for spk in range(num_spk):
    # Opening JSON file
        file = "/dsi/gannot-lab/datasets/mordehay/data_for_robot_one_speaker/test/labels/scenario_{0}_spk_{1}.json".format(num, spk) #need to change!!
        with open(file) as f:
            # returns JSON object as
            # a dictionary
            data = json.load(f)
        for segment in data["speech_segments"]:
            vad[spk, segment["start_time"]:segment["end_time"]] = 1
        if save_each_spk:
            vad_spk = vad[spk, :]
            vad_frames[spk, 1:-1] = convert_samples2frames_vad(vad_spk)
            #for the begining edge
            count_array = np.bincount(np.concatenate((np.zeros(256, dtype=np.int64),vad_spk[:256])))
            vad_frames[spk, 0] = np.argmax(count_array)
            #for the final edge
            count_array = np.bincount(np.concatenate((vad_spk[-256:], np.zeros(256, dtype=np.int64))))
            vad_frames[spk, -1] = np.argmax(count_array)
    vad_sum = np.sum(vad, axis=0)
    num_frames = int(2 * cut_length/args["nfft"] - 1 + 2)  #len(samples)/512 + (len(samples)/512 - 1) + 2 (the 2 frames are the edges)
    vad_frames_sum = np.zeros(num_frames)
    vad_frames_sum[1:-1] = convert_samples2frames_vad(vad_sum)
    #for the begining edge
    count_array = np.bincount(vad_sum[:256])
    vad_frames_sum[0] = np.argmax(count_array)
    #for the final edge
    count_array = np.bincount(vad_sum[-256:])
    vad_frames_sum[-1] = np.argmax(count_array)
    if save_each_spk:
        np.savez(os.path.join(save_path, "scenario_{0}.npz".format(num)), vad_frames_sum = vad_frames_sum, vad_frames_individual = vad_frames)
    else:
        np.savez(os.path.join(save_path, "scenario_{0}.npz".format(num)), vad_frames_sum = vad_frames_sum)    

#pprint(data)


 