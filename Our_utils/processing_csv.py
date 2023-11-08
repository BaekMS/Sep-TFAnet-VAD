import pandas as pd
import numpy as np
import os

path = "/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/train/csv_files/with_wham_noise_res.csv" ###need to change
df = pd.read_csv(path)

##add drr
gamma = 1
df["drr"] = 0.057 * np.sqrt((gamma * np.array(df["room_x"]) * np.array(df["room_y"]) * np.array(df["room_z"])) / np.array(df["rt60"]))



##calc distance of each speaker from the ref mic
spk1_loc = np.stack([df["speaker1_x"], df["speaker1_y"], df["speaker1_z"]], axis=1)
spk2_loc = np.stack([df["speaker2_x"], df["speaker2_y"], df["speaker2_z"]], axis=1)
mic_loc = np.stack([df["mic1_x"], df["mic1_y"], df["mic1_z"]], axis=1)
df["distance_spk0_mic0"] = np.linalg.norm(spk1_loc - mic_loc, axis=1)
df["distance_spk1_mic0"] = np.linalg.norm(spk2_loc - mic_loc, axis=1)

##whether the distnace of each speaker is greater than the drr
df["speaker0_far_drr"] = list(map(lambda bol: int(bol), list(df["distance_spk0_mic0"] > df["drr"])))
df["speaker1_far_drr"] = list(map(lambda bol: int(bol), list(df["distance_spk1_mic0"] > df["drr"])))

##read spekaer gender file
with open("Gender.txt", "r") as file_gender:
    list_gender_speaker = file_gender.read().splitlines()
    gender_speaker_dict = dict(list(map(lambda sp_ge: tuple([int(sp_ge.split(',')[0]), int(sp_ge.split(',')[1])]), list_gender_speaker)))


##add speaker gender

df["speaker0_gender"] = list(map(lambda spk_id: gender_speaker_dict[spk_id], df["speaker1_id"]))#zero for female and one for male
df["speaker1_gender"] = list(map(lambda spk_id: gender_speaker_dict[spk_id], df["speaker2_id"]))

#add gender

df["same_gender"] = 1 - (df["speaker0_gender"] ^ df["speaker1_gender"]) #same gender is '1', otherwise '0'




print(os.path.splitext(path)[0] +  '_more_informative.csv')
new_path = os.path.splitext(path)[0] +  '_more_informative.csv'
##save processing
df.to_csv(new_path)

