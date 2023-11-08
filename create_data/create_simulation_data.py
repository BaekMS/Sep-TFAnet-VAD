# Author: Mordehay And Renana


from tqdm import tqdm
import argparse
import random
from random import seed
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import csv
import os
from os import listdir
from pathlib import Path
import glob
import yaml
import pyrirgen
from scipy.io import wavfile


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lab/renana/PycharmProjects/AV_rtf_separation/create_data_frenkel
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lab/renana/PycharmProjects/rir_gen_new/

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

# We add
def rotation_matrix(theta, mat):
    """

    Args:
        theta: how much degrees to rotate when the the rotation is respect to the X-axis and counterclockwise
        mat: matrix to be rotated with shape [3, m] when m is any real number

    Returns: rotated matrix with shape [3, m]

    """
    Rotation_matrix = np.array([
        [math.cos(math.radians(theta)), -math.sin(math.radians(theta)), 0],
        [math.sin(math.radians(theta)), math.cos(math.radians(theta)), 0],
        [0, 0, 1]])
    rotated_matrix = Rotation_matrix @ mat
    return rotated_matrix


class Room:
    def __init__(self, args):
        super().__init__()

        min_w = args.min_w
        max_w = args.max_w
        min_l = args.min_l
        max_l = args.max_l
        min_h = args.min_h
        max_h = args.max_h
        # min_rt60 = 0.2 # less than 0.13 create NaN
        max_rt60 = args.max_rt60
        min_rt60 = args.min_rt60

        self.room_width = np.random.uniform(min_w, max_w)
        self.room_length = np.random.uniform(min_l, max_l)
        self.room_height = np.random.uniform(min_h, max_h)
        if args.not_rand_rt60:
            self.rt60 = args.rt60
        else:
            self.rt60 = np.random.uniform(min_rt60, max_rt60)
        # print(self.rt60)

        # print('Room shape is {0:.2f}x{1:.2f}x{2:.2f}'.format(self.room_width, self.room_length, self.room_height))

    def get_dims(self):
        return round(self.room_width, 2), round(self.room_length, 2), round(self.room_height, 2), self.rt60


class Array:
    def __init__(self, width, length, array_area_w=0.2, array_area_l=0.2, res=5):
        super().__init__()
        # array_area_w=0.5 ## add to configuration file
        # res=5 ## add to configuration file
        min_array_z = args.min_array_z
        max_array_z = args.max_array_z

        self.array_x = np.random.uniform(0.5 * width - array_area_w, 0.5 * width + array_area_w)
        self.array_y = np.random.uniform(0.5 * length - array_area_l, 0.5 * length + array_area_l)
        self.array_z = round(np.random.uniform(min_array_z, max_array_z), 2)
        self.n_mic = args.n_mics

        theta_opt = np.arange(0, 360, res)  ## add to configuration file #doesn't include 360!
        theta_inx = np.random.randint(len(theta_opt))  # [ 0, len(thehta_opt) )
        self.array_theta = theta_opt[theta_inx]
        self.array_type = args.array_type
        self.mic_loc = args.mic_loc
        # print('Array center was located in ({0:.2f},{1:.2f},{2:.2f}) with theta = {3}'
        #     .format(self.array_x, self.array_y, self.array_z, self.array_theta))

    def get_array_loc(self):
       
        temp = rotation_matrix(-90,
                                np.array(self.mic_loc).T)  # make the glasses to be aligned with respect to X-axis
        receivers = rotation_matrix(self.array_theta, temp) + np.array(
            [[self.array_x, self.array_y, self.array_z]]).T  # shape = [3, num_of_mic]

        return self.array_x, self.array_y, self.array_z, self.array_theta, receivers.T


class Speakers:
    def __init__(self, args, max_speakers, width, length, height, array_x, array_y, array_z, array_theta, rt60,
                 limit=0.5):
        super().__init__()
        self.args = args
        if args.not_rand_speak:
            self.N = args.n_speakers
        else:
            self.N = np.random.randint(1, max_speakers + 1)
        self.width = width
        self.length = length
        self.height = height
        self.rt60 = rt60
        self.array_x = array_x
        self.array_y = array_y
        self.array_z = array_z
        self.array_theta = array_theta
        self.n_mic = self.args.n_mics
        self.mum_channel = np.array(args.mic_loc).shape[0]
        self.limit = args.limit_speaker_dis

    def find_r_theta(self, theta_opt):
        theta_inx = np.random.randint(len(theta_opt))
        speaker_theta = theta_opt[theta_inx]
        if speaker_theta < 180 and speaker_theta != 0 and speaker_theta != 90:
            y_limit = self.length - self.limit - self.array_y
            x_max = y_limit / (math.tan(math.radians(speaker_theta)))
            if speaker_theta < 90:
                x_limit = self.width - self.limit - self.array_x
            elif speaker_theta > 90:
                x_limit = -(self.array_x - self.limit)
            y_max = math.tan(math.radians(speaker_theta)) * x_limit
            if speaker_theta < 90:
                max_speaker_x = min([x_max, x_limit])
            elif speaker_theta > 90:
                max_speaker_x = max([x_max, x_limit])
            max_speaker_y = min([y_max, y_limit])
            max_speaker_r = math.sqrt(pow(max_speaker_x, 2) + pow(max_speaker_y, 2))
        elif speaker_theta > 180 and speaker_theta != 270:
            y_limit = -(self.array_y - self.limit)
            x_max = y_limit / (math.tan(math.radians(speaker_theta - 180)))
            if speaker_theta < 270:
                x_limit = -(self.array_x - self.limit)
            elif speaker_theta > 270:
                x_limit = self.width - self.limit - self.array_x
            y_min = math.tan(math.radians(speaker_theta - 180)) * x_limit
            if speaker_theta < 270:
                max_speaker_x = max([x_max, x_limit])
            elif speaker_theta > 270:
                max_speaker_x = min([x_max, x_limit])
            max_speaker_y = max([y_min, y_limit])
            max_speaker_r = math.sqrt(pow(max_speaker_x, 2) + pow(max_speaker_y, 2))
        elif speaker_theta == 0:
            max_speaker_r = self.width - self.limit - self.array_x
        elif speaker_theta == 90:
            max_speaker_r = self.length - self.limit - self.array_y
        elif speaker_theta == 180:
            max_speaker_r = self.array_x - self.limit
        elif speaker_theta == 270:
            max_speaker_r = self.array_y - self.limit


        if max_speaker_r > args.mic2speaker_max:
            max_speaker_r = args.mic2speaker_max
        speaker_r = np.random.uniform(args.mic2speaker_min, max_speaker_r)

        speaker_x = self.array_x + speaker_r * math.cos(math.radians(speaker_theta))
        speaker_y = self.array_y + speaker_r * math.sin(math.radians(speaker_theta))
        speaker_z = round(np.random.uniform(args.min_array_z, args.max_array_z), 2)

        if self.array_theta <= speaker_theta:
            speaker_theta_array = speaker_theta - self.array_theta
        else:
            speaker_theta_array = 360 - (self.array_theta - speaker_theta)

        return round(speaker_r, 2), speaker_theta, speaker_theta_array, \
               round(speaker_x, 2), round(speaker_y, 2), speaker_z

    def choose_wham(self, wham_path_folder, cut_length):

        wav_len = 0
        while wav_len < cut_length:
            wham_list = listdir(wham_path_folder)
            wham_wav = random.choices(wham_list)[0]
            wham_path = wham_path_folder + '/' + wham_wav
            fs, wave = wavfile.read(wham_path)
            if self.args.wham:
                wave = np.transpose(wave[: cut_length, 0])
                wav_len = len(wave)
            if self.args.bubble:
                wave = np.transpose(wave[: cut_length, :])
                wav_len = len(wave[0, :])
                # print(wav_len)

        return wave

    def get_first_speaker(self, res=0.5):
        res = args.res
        theta_opt = np.arange(0, 180, res)
        speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z = self.find_r_theta(theta_opt)

        return speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z

    def get_speaker(self, angles, i, islast, res=5, speaker_dist=20):
        res = args.res
        speaker_dist = args.speaker_dist
        theta_opt = np.arange(0, 180, res)
        for a in angles:
            del_inx = np.where(((theta_opt < a + speaker_dist) & (theta_opt > a - speaker_dist))
                               | (theta_opt > 360 + a - speaker_dist) | (theta_opt < a - 360 + speaker_dist))
            theta_opt = np.delete(theta_opt, del_inx)
        speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z = self.find_r_theta(theta_opt)


        return speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z

    def get_speakers(self, mics_loc, i):
        r1, theta1, theta1_array, x1, y1, z1 = self.get_first_speaker()
        speakers_loc = []
        thetas = []
        s_thetas_array = []
        dists = []
        speakers_loc.append([x1, y1, z1])
        thetas.append(theta1)
        s_thetas_array.append(theta1_array)
        dists.append(r1)
        is_last = False
        for s in range(1, self.N):
            if s == self.N - 1:
                is_last = True
            r, theta, theta_array, x, y, z = self.get_speaker(thetas, s + 1, is_last)
            thetas.append(theta)
            s_thetas_array.append(theta_array)
            dists.append(r)
            speakers_loc.append([x, y, z])


        h_list, h_list_1_order = self.get_speakers_h(speakers_loc, mics_loc)
        speakers_id, speakers_path = self.get_speakers_id()
        speakers_wav_files_p = self.choose_wav(speakers_path)
        speakers_pass2_conv_wav_h = speakers_wav_files_p[0]
        speakers_conv, _, mic_snr, sir = self.conv_wav_h(speakers_pass2_conv_wav_h, h_list, i, dists, h_list_1_order,
                                                    s_thetas_array)
        if self.args.partial_overlap:
            speakers_path_clean = speakers_wav_files_p[1]
        else:
            speakers_path_clean = speakers_wav_files_p[0]
        return self.N, dists, s_thetas_array, speakers_loc, speakers_id, h_list, speakers_conv, speakers_path_clean, \
               mic_snr, sir, self.overlap_ratio

    def get_speaker_h(self, speaker_x, speaker_y, speaker_z, receivers):
        room_measures = [self.width, self.length, self.height]
        source_position = [speaker_x, speaker_y, speaker_z]
        fs = args.fs
        n = args.n
        # creat a dict that will be added to h_df
        h_df_sample = {'room_measures': room_measures, 'source_position': source_position, 'receivers': receivers,
                       'reverbTime': self.rt60, 'fs': fs, 'orientation': [0, .0], 'nSamples': n, 'nOrder': -1}
        # print(h_df_sample)
        # original rir
        '''
        Args of pyrirgen:
            c (float): sound velocity in m/s
            fs (float): sampling frequency in Hz
            receiverPositions (list[list[float]]): M x 3 array specifying the (x,y,z) coordinates of the receiver(s) in m
            sourcePosition (list[float]): 1 x 3 vector specifying the (x,y,z) coordinates of the source in m
            roomMeasures (list[float]): 1 x 3 vector specifying the room dimensions (x,y,z) in m
            betaCoeffs (list[float]): 1 x 6 vector specifying the reflection coefficients [beta_x1 beta_x2 beta_y1 beta_y2 beta_z1 beta_z2]
            reverbTime (float): reverberation time (T_60) in seconds
            nSample (int): number of samples to calculate, default is T_60*fs
            micType (str): [omnidirectional, subcardioid, cardioid, hypercardioid, bidirectional], default is omnidirectional
            nOrder (int): reflection order, default is -1, i.e. maximum order
            nDim (int): room dimension (2 or 3), default is 3
            orientation (list[float]): direction in which the microphones are pointed, specified using azimuth and elevation angles (in radians), default is [0 0]
            isHighPassFilter (bool)^: use 'False' to disable high-pass filter, the high-pass filter is enabled by default.
        '''
        h = pyrirgen.generateRir(roomMeasures=room_measures, sourcePosition=source_position,
                                 receiverPositions=receivers, reverbTime=self.rt60,
                                 fs=fs, orientation=[0, .0], nSamples=n, nOrder=-1)
        h_1_order = pyrirgen.generateRir(roomMeasures=room_measures, sourcePosition=source_position,
                                         receiverPositions=receivers, reverbTime=0,
                                         fs=fs, orientation=[0, .0], nSamples=n, nOrder=-1)

        return np.array(h), np.array(h_1_order), h_df_sample

    def get_speakers_h(self, speakers_loc, mics_loc):

        speakers_h = []
        speakers_h_1_order = []

        for inx, speaker in enumerate(speakers_loc):
            h, h_1_order, h_sim = self.get_speaker_h(speaker[0], speaker[1], speaker[2], mics_loc)
            speakers_h.append(h)
            speakers_h_1_order.append(h_1_order)

        return speakers_h, speakers_h_1_order

    def get_speakers_id(self):
        if self.args.mode == 'train' or self.args.mode == 'val':
            PATH = self.args.s_path + '/Train'
            ids = listdir(PATH)
        elif self.args.mode == 'test':
            PATH = self.args.s_path + '/Test'
            ids = listdir(PATH)
        else:
            raise Exception("Chosen mode is not valid!!")
        id_list = []
        id_list.append('spk')
        id_list.append('spk')
        
        while id_list[0] == id_list[1]:
            id_list = random.choices(ids, k=self.N)
        id_path = [os.path.join(PATH, id) for id in id_list]
        if len(id_list) > self.N:
            print('more id than N')

        return id_list, id_path

    def unique(self, list):

        unique_flag = True
        not_unique_list = []
        unique_list = []
        for element in list:
            # check if exists in unique_list or not
            if element not in unique_list:
                unique_list.append(element)
            else:
                not_unique_list.append(element)
                # print(element, 'is exist twice')
        if len(not_unique_list):
            unique_flag = False
        # print list
        # for x in unique_list:
        #   print('sentences list', x)
        return unique_flag

    def check_wav_len(self, wav_files, cut_length, wav_lists1):
        if self.args.partial_overlap:
            # overlap_ratio = random.choices([20, 40, 60, 80, 100], weights=(20, 20, 20, 20, 20), k=1)[0] #return list and hench the zero index
            overlap_ratio = random.choices(args.overlap_ratio,\
             weights=args.overlap_ratio_weights, k=1)[0]  # return list and hench the zero index

            self.overlap_ratio = overlap_ratio
            # print(overlap_ratio)
            overlap_minutes = overlap_ratio / 100 * self.args.aud_length
            overlap_samples = int(overlap_minutes * self.args.fs)
            overlap_samples_len = overlap_samples
            waves = []
            num_spekaers = 2  # need to change the code for more than 2 speakers
            waves_path_clean = []
            aud_length_waves = np.zeros(shape=(num_spekaers, self.args.aud_length * self.args.fs))
            if overlap_ratio <= 40:
                overlap_samples_len = 2 * overlap_samples #TODO
            for i, wave_file in enumerate(wav_files):
                fs, wave = wavfile.read(wave_file)
                while len(wave) < overlap_samples_len:
                    wav_files[i] = random.choice(wav_lists1[i])
                    fs, wave = wavfile.read(wav_files[i])

                waves_path_clean.append(wav_files[i])
                final_sample = np.random.choice(np.arange(overlap_samples_len, self.args.aud_length * self.args.fs + 1))
                waves.append(wave[:final_sample])  # truncated waves

            is_subset = random.choices([True, False], weights=args.subset_ratio_weights, k=1)[0]
            begin_sample_wave0 = np.random.choice(np.arange(0, self.args.aud_length * self.args.fs + 1 - len(waves[0])))
            final_sample_wave0 = begin_sample_wave0 + len(waves[0])
            aud_length_waves[0][begin_sample_wave0: final_sample_wave0] = waves[0]
            if is_subset:
                # print("Subset")
                waves[1] = waves[1][:overlap_samples]  # this is the short one (wave[0] is the long one)
                begin_sample_wav1 = np.random.choice(
                    np.arange(begin_sample_wave0, final_sample_wave0 - len(waves[1]) + 1))
                aud_length_waves[1][begin_sample_wav1: begin_sample_wav1 + len(waves[1])] = waves[1]
            else:
                is_in_begining = random.choices([True, False], weights=(50, 50), k=1)[0]
                if is_in_begining:
                    # print("begining")
                    final_sample_wav1 = begin_sample_wave0 + overlap_samples
                    begin_sample_wav1 = max(np.random.choice(np.arange(0, begin_sample_wave0 + 1)),
                                            final_sample_wav1 - len(waves[1]))
                    new_len_wav1 = final_sample_wav1 - begin_sample_wav1
                    aud_length_waves[1][begin_sample_wav1: final_sample_wav1] = waves[1][len(waves[1]) - new_len_wav1:]
                else:
                    # print("final")
                    begin_sample_wav1 = final_sample_wave0 - overlap_samples
                    final_sample_wav1 = min(
                        np.random.choice(np.arange(final_sample_wave0, self.args.aud_length * self.args.fs + 1)),
                        begin_sample_wav1 + len(waves[1]))
                    new_len_wav1 = final_sample_wav1 - begin_sample_wav1
                    aud_length_waves[1][begin_sample_wav1: final_sample_wav1] = waves[1][:new_len_wav1]


            return (list(aud_length_waves), waves_path_clean)

        else:
            for i, wave_file in enumerate(wav_files):
                fs, wave = wavfile.read(wave_file)
                while len(wave) < cut_length:
                    wav_files[i] = random.choice(wav_lists1[i])
                    fs, wave = wavfile.read(wav_files[i])

            return wav_files

    def choose_wav(self, id_path):

        spk1 = 'spk'
        spk2 = 'spk'

        while spk1 == spk2:

            wav_lists = []
            fs = self.args.fs
            num_frames = int(np.floor((args.aud_length * fs) / args.nfft))
            cut_length = int(num_frames * args.nfft)
            # print(f"id path is: {id_path}")
            if self.args.dataset_name == 'librispeech':
                #print(id_path)
                for path in id_path:
                    wav_lists.append(glob.glob(path + '/*/*.wav'))
                # print(wav_lists)
                wav_lists1 = []
                for wav_list in wav_lists:
                    wav_lists1.append([path for path in wav_list if os.path.splitext(path)[-1] == '.wav'])
                wav_files = [random.choice(wav_list) for wav_list in wav_lists1]  ###BE CAREFUL

            elif self.args.dataset_name == 'wsj':
                for path in id_path:
                    wav_lists.append([os.path.join(path, wav_file) for wav_file in listdir(path)])

                unique_flag = False
                while not unique_flag:
                    wav_lists1 = []
                    for wav_list in wav_lists:
                        wav_lists1.append([path for path in wav_list if os.path.splitext(path)[1] == '.wav'])
                    wav_files = [random.choice(wav_list) for wav_list in wav_lists1]  ###BE CAREFUL

                    len(wav_files)
                    sentences = [wav_string[-9:-4] for wav_string in wav_files]  ###BE CAREFUL-only for wsj0!!
                    unique_flag = self.unique(sentences)

            # wav_files
            if self.args.dataset_name == 'librispeech':
                spk1 = wav_files[0].split('/')[-3]
                spk2 = wav_files[1].split('/')[-3]
                if spk1 == spk2:
                    print('same speaker')

            elif self.args.dataset_name == 'wsj':
                spk1 = wav_files[0][-16:-13]
                spk2 = wav_files[1][-16:-13]
                if spk1 == spk2:
                    print('same speaker')
            wav_files = self.check_wav_len(wav_files, cut_length, wav_lists1)
        return wav_files

    def conv_wav_h(self, wav_files, h_list, i, dists, h_list_1_order, s_thetas_array):  # i = scenario number
        speakers_conv = []
        speakers_delay_conv = []

        for wave_file, h, h_1_order in zip(wav_files, h_list, h_list_1_order):

            if self.args.partial_overlap:
                fs = self.args.fs
                wave = wave_file
            else:
                fs, wave = wavfile.read(wave_file)
                wave = np.copy(wave)
            num_frames = int(np.floor((args.aud_length * fs) / args.nfft))
            cut_length = int(num_frames * args.nfft)

            speaker_conv = []
            speaker_delayed_conv = []
            if h.ndim == 1:
                wav_h_conv_all = np.convolve(wave, h)
                wav_h_conv = wav_h_conv_all[: cut_length] 
                speaker_conv.append(np.expand_dims(wav_h_conv, axis=0))
            else:
                for s_mic_h in h:
                    wav_h_conv_all = np.convolve(wave, s_mic_h)
                    wav_h_conv = wav_h_conv_all[: cut_length]
                    speaker_conv.append(np.expand_dims(wav_h_conv, axis=0))

            speaker_conv_np = np.concatenate(speaker_conv)
            speakers_conv.append(speaker_conv_np)

            if h_1_order.ndim == 1:
                h_delay = h_1_order
            else:
                h_delay = h_1_order[0]

            wave_h_delay_conv_all = np.convolve(wave, h_delay)
            wave_h_delay_conv = wave_h_delay_conv_all[: cut_length]
            speaker_delayed_conv.append(np.expand_dims(wave_h_delay_conv, axis=0))
            speaker_delayed_conv_np = np.concatenate(speaker_delayed_conv)
            speakers_delay_conv.append(speaker_delayed_conv_np)


        len_list = [arr.shape[1] for arr in speakers_conv]
        min_len = min(len_list)
        if min_len < num_frames * args.nfft:
            args.short_sam = args.short_sam + 1
            zero_pad = np.int(cut_length - min_len)

            speakers_conv = [arr[:, :min_len] for arr in speakers_conv]
            speakers_conv_cut = [np.concatenate((arr, np.zeros((arr.shape[0], np.int(zero_pad)))), axis=1) for arr in
                                 speakers_conv]
        else:
            speakers_conv_cut = [arr[:, :cut_length] for arr in speakers_conv]
        directional_noise = 0
        mic_noise = 0
        wham_noise = 0


        if self.args.add_mic_noise:
            mic_noise = np.random.normal(loc=0, scale=1, size=(self.mum_channel, cut_length))  # mic_noise
            mic_snr = np.random.uniform(args.mic_snr_l, args.mic_snr_h)
            mic_noise = Noise(self.args).get_mixed(mixed_sig_np, mic_noise, mic_snr)  # mic_noise after update of snr
            snr = mic_snr

        if self.args.wham:
            sir = np.random.uniform(args.min_sir, args.max_sir)#0  ##can be random

            #print(speakers_conv_cut)
            mixed_sig_np = speakers_conv_cut[0]#sum(speakers_conv_cut) #####just for one speaker!!
            #print(f"**************** {len(mixed_sig_np)}")
            noise = Noise(self.args)
            wham_path_folder = noise.choose_noise_path()
            # get_mixed(self, mixed, noise, snr)
            # print('wham_path_folder', wham_path_folder)
            wham_noise = self.choose_wham(wham_path_folder, cut_length)  # mic_noise
            wham_snr = np.random.uniform(args.wham_snr_l, args.wham_snr_h)
            #wham_snr = sir#np.random.uniform(low=sir, high=sir+1)
            """wham_noise = Noise(self.args).get_mixed(speakers_conv_cut[0][0, :], wham_noise,
                                                    wham_snr)  # mic_noise after update of snr"""
            wham_noise = Noise(self.args).get_mixed(mixed_sig_np[0,:], wham_noise, wham_snr)                                        
            snr = wham_snr
            #print(f"snr = {snr} \t sir = {sir}")
        elif self.args.bubble:
            mixed_sig_np = sum(speakers_conv_cut)
            noise = Noise(self.args)
            bubble_path_folder = noise.choose_noise_path()
            # get_mixed(self, mixed, noise, snr)
            # print('bubble_path_folder', bubble_path_folder)
            bubble_noise = self.choose_wham(bubble_path_folder, cut_length)  # mic_noise
            bubble_snr = np.random.uniform(args.mic_snr_l, args.mic_snr_h)
            bubble_noise = Noise(self.args).get_mixed(mixed_sig_np, bubble_noise,
                                                      bubble_snr)  # mic_noise after update of snr
            snr = bubble_snr
        else:
            mixed_sig_np = sum(speakers_conv_cut)

        if self.args.wham:
            mixed_all = mixed_sig_np.copy()
            mixed_all[0, :] = mixed_all[0, :] + wham_noise #####just for one speaker!!
        elif self.args.bubble:
            mixed_all = mixed_sig_np.copy()
            mixed_all = mixed_all + bubble_noise
        else:
            mixed_all = mixed_sig_np + directional_noise + mic_noise
            mixed_sig_np = mixed_sig_np + directional_noise + mic_noise

        '''
        #save an example
        from scipy.io.wavfile import write
        to_save = mixed_sig_np[[4, 5], :]
        to_save = to_save.T
        write("without_noise.wav", 16000, to_save.astype(np.int16))
        mixed_sig_np = mixed_sig_np + mic_noise + directional_noise

        to_save = mixed_sig_np[[4,5],:]
        to_save = to_save.T
        write("with_white_noise.wav", 16000, to_save.astype(np.int16))
        '''
        ## Normalizing result to between -1 and 1
        max_conv_val = np.max(mixed_sig_np)
        min_conv_val = np.min(mixed_sig_np)
        mixed_sig_np = 2 * (mixed_sig_np - min_conv_val) / (
                    max_conv_val - min_conv_val) - 1  # normalize between -1 and 1
        if self.args.wham:
            max_conv_val = np.max(mixed_all[0, :])
            min_conv_val = np.min(mixed_all[0, :])
            mixed_all[0, :] = 2 * (mixed_all[0, :] - min_conv_val) / (
                    max_conv_val - min_conv_val) - 1  # normalize between -1 and 1
        elif self.args.bubble:
            max_conv_val = np.max(mixed_all)
            min_conv_val = np.min(mixed_all)
            mixed_all = 2 * (mixed_all - min_conv_val) / (
                    max_conv_val - min_conv_val) - 1  # normalize between -1 and 1

        idx = 0

        speakers_target = []
        while idx <= len(speakers_conv_cut) - 1:
            speaker_sig = speakers_conv_cut[idx]
            # print(speaker_sig.shape)
            # Normalizing target to between -1 and 1
            max_conv_val = np.max(speaker_sig)
            min_conv_val = np.min(speaker_sig)
            speaker_sig = 2 * (speaker_sig - min_conv_val) / (max_conv_val - min_conv_val) - 1
            speakers_target.append(speaker_sig)
            idx += 1

        speakers_delay = []
        speakers_delay_conv_cut = [arr[:, :cut_length] for arr in speakers_delay_conv]
        idx_d = 0
        while idx_d <= len(speakers_delay_conv_cut) - 1:
            speaker_sig_delay = speakers_delay_conv_cut[idx_d]
            # print(speaker_sig.shape)
            # Normalizing target to between -1 and 1
            max_conv_val = np.max(speaker_sig_delay)
            min_conv_val = np.min(speaker_sig_delay)
            speaker_sig_delay = 2 * (speaker_sig_delay - min_conv_val) / (max_conv_val - min_conv_val) - 1
            speakers_delay.append(speaker_sig_delay)
            idx_d += 1
        speakers_delay = np.concatenate((speakers_delay[0], speakers_delay[1]), axis=0)

        # speakers_delay_np=np.asarray(speakers_delay)

        conv_dir = os.path.join(args.results_path, args.mode, args.which_noise + '_audio')
        # print('conv_dir', conv_dir)
        conv_path = os.path.join(conv_dir, 'scenario_{0}.p'.format(i))
        Path(conv_dir).mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(conv_dir):
        #    os.mkdir(conv_dir)

        if self.args.doa:
            with open(conv_path, 'wb') as f:
                # print('mixed_sig_np shape', np.float32(mixed_sig_np).shape)
                # print('speakers_target shape', len(speakers_target))
                # print('speakers_target shape,0', speakers_target[0].shape)
                # print('s_thetas_array shape', len(s_thetas_array))
                temp = np.array(speakers_target, dtype=np.float32)
                clean_speakers = temp[:, 0, :]
                if self.args.wham:
                    noisy_signal = np.float32(mixed_all[0, :])
                elif self.args.bubble:
                    noisy_signal = np.float32(mixed_all)
                else:
                    noisy_signal = 0

                '''
                plt.figure()
                plt.plot(np.arange(0, len(clean_speakers[0, :])), clean_speakers[0, :])
                plt.title('clean0')
                plt.show()
                plt.figure()
                plt.plot(np.arange(0, len(clean_speakers[1, :])), clean_speakers[1, :])
                plt.title('clean1')
                plt.show()

                plt.figure()
                plt.plot(np.arange(0, len(speakers_delay[0,:])), speakers_delay[0,:])
                plt.title('speakers_delay0')
                plt.show()

                plt.figure()
                plt.plot(np.arange(0, len(speakers_delay[1, :])), speakers_delay[1, :])
                plt.title('speakers_delay1')
                plt.show()
                '''
                #speakers_delay = np.zeros_like(speakers_delay) #####just for one speaker!!
                #clean_speakers = np.zeros_like(clean_speakers) #####just for one speaker!!
                speakers_delay[1] = np.zeros_like(speakers_delay[0]) #####just for one speaker!!
                clean_speakers[1] = np.zeros_like(clean_speakers[0]) #####just for one speaker!!
                pickle.dump((np.float32(mixed_sig_np), np.float32(noisy_signal),
                             clean_speakers, speakers_delay, np.array(s_thetas_array, dtype=np.float32)), f)

        else:
            with open(conv_path, 'wb') as f:
                pickle.dump((mixed_sig_np, speakers_delay), f)
            # wave_path = os.path.join(conv_dir, 'scenario_{0}_mixed.wav'.format(i + 1))
            # wavfile.write(wave_path, fs, mixed_sig_np)
            # wave_path = os.path.join(conv_dir, 'scenario_{0}_cln.wav'.format(i + 1))
            # wavfile.write(wave_path, fs, speakers_delay[0])
        return mixed_sig_np, conv_path, snr, sir


class Noise:
    def __init__(self, args):
        super().__init__()

        # self.snr = np.random.uniform(5, 15)
        # self.mic_snr = args.mic_snr # 20
        self.directional_snr = args.directional_snr
        self.args = args

    def choose_noise_path(self):
        if self.args.mode == 'train':
            path = self.args.n_path + '/tr'
        elif self.args.mode == 'val':
            path = self.args.n_path + '/cv'
        elif self.args.mode == 'test':
            path = self.args.n_path + '/tt'
        else:
            raise Exception("Chosen mode is not valid!!")

        return path

    def get_mixed(self, mixed, noise, snr):
        # snr = 0
        mix_std = np.std(mixed)
        noise_std = np.std(noise)
        noise_gain = np.sqrt(10 ** (-snr / 10) * np.power(mix_std, 2) / np.power(noise_std, 2))
        noise = noise_gain * noise
        return noise

def plot_room(args, width, length, height, speakers_loc, mic_xy, scenario):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Scenario {0}'.format(scenario))
    ax.set_xlim(0, width)
    ax.set_ylim(0, length)
    ax.set_zlim(0, height)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    speakers_loc_np = np.array(speakers_loc)
    mic_xy_np = np.array(mic_xy)
    speakers_x = speakers_loc_np[:, 0]
    speakers_y = speakers_loc_np[:, 1]
    speakers_z = speakers_loc_np[:, 2]
    mics_x = mic_xy_np[:, 0]
    mics_y = mic_xy_np[:, 1]
    mics_z = mic_xy_np[:, 2]

    ax.scatter(speakers_x, speakers_y, speakers_z, marker='o', s=20, c='b', depthshade=True)
    ax.scatter(mics_x, mics_y, mics_z, marker='x', s=20, c='r', depthshade=True)
    plt.savefig('{0}/scenario_{1}/room.png'.format(args.results_path, scenario + 1))


def get_scenario_data(args, scenario):
    room = Room(args)
    w, l, h, rt60 = room.get_dims()
    array = Array(w, l)
    array_x, array_y, array_z, array_theta, mics_xy = array.get_array_loc()
    max_speakers = args.max_speakers
    speakers = Speakers(args, max_speakers, w, l, h, array_x, array_y, array_z, array_theta, rt60)
    num_speakers, dists, s_thetas_array, speakers_xy, speakers_id, h_list, conv_speakers, speakers_wav_files, mic_snr, \
    sir, overlap_ratio = speakers.get_speakers(mics_xy, scenario)
    # plot_room(args, w, l, h, speakers_xy, mics_xy, scenario)
    return w, l, h, rt60, array_theta, mics_xy, speakers_xy, dists, s_thetas_array, speakers_id, h_list, \
           conv_speakers, [array_x, array_y, array_z], speakers_wav_files, mic_snr, sir, overlap_ratio


def create_csv_results(args):
    csv_path = os.path.join(args.results_path, args.mode, 'csv_files')
    Path(csv_path).mkdir(parents=True, exist_ok=True)

    # write to csv file name with 1
    with open('{0}_res.csv'.format(os.path.join(csv_path, args.which_noise)), 'w', newline='') as file:
        # print(file)
        #first_row = ['scenario', 'room_x', 'room_y', 'room_z', 'rt60', 'mic_snr', 'sir', 'overlap_ratio']
        #first_row.append('num_mics')
        first_row = ['scenario', 'room_x', 'room_y', 'room_z', 'rt60', 'mic_snr', 'sir', 'overlap_ratio', 'num_mics',
                     'array_theta','num_speakers', 'path_file' ]
        dims = ['x', 'y', 'z']
        for dim in ['x', 'y', 'z']:
            first_row.append('array_{0}'.format(dim))
        for mic in range(args.n_mics):
            for dim in dims:
                first_row.append('mic{0}_{1}'.format(mic + 1, dim))
        #first_row.append('array_theta')
        #first_row.append('num_speakers')
        for s in range(args.max_speakers):
            first_row.append('speaker{0}_id'.format(s + 1))
            first_row.append('speaker{0}_radius'.format(s + 1))
            first_row.append('speaker{0}_doa'.format(s + 1))
            for dim in dims:
                first_row.append('speaker{0}_{1}'.format(s + 1, dim))
            first_row.append("speaker{0}_path_clean".format(s))
        #first_row.append("path_file")

        writer = csv.DictWriter(file, fieldnames=first_row)
        writer.writeheader()

        for result in tqdm(range(args.num_results)):
            w, l, h, rt60, array_theta, mics_xy, speakers_xy, dists, thetas, speakers_id, _, _, array_loc, \
            speakers_wav_files, mic_snr, sir, overlap_ratio = get_scenario_data(args, result)

            row_dict = {'scenario': result, 'room_x': w, 'room_y': l, 'room_z': h, 'rt60': round(rt60, 2),
                        'mic_snr': round(mic_snr, 2), 'sir': round(sir, 2), 'overlap_ratio': overlap_ratio}
            mics_xy_np = np.array(mics_xy)
            row_dict['num_mics'] = args.n_mics
            row_dict["path_file"] = os.path.join(args.results_path, args.mode, args.which_noise + '_audio',
                                                 'scenario_{0}.p'.format(result))
            for i, dim in enumerate(['x', 'y', 'z']):
                row_dict['array_{0}'.format(dim)] = array_loc[i]
            for mic in range(args.n_mics):
                for dim_inx, dim in enumerate(dims):
                    if mics_xy_np.ndim == 1:
                        row_dict['mic{0}_{1}'.format(mic + 1, dim)] = mics_xy_np[dim_inx]
                    else:
                        row_dict['mic{0}_{1}'.format(mic + 1, dim)] = mics_xy_np[mic][dim_inx]

            for i, path in enumerate(speakers_wav_files):
                row_dict["speaker{0}_path_clean".format(i)] = path

            row_dict['array_theta'] = array_theta
            speakers_xy_np = np.array(speakers_xy)
            row_dict['num_speakers'] = len(speakers_xy)
            for s in range(len(speakers_xy)):
                row_dict['speaker{0}_id'.format(s + 1)] = speakers_id[s]
                row_dict['speaker{0}_radius'.format(s + 1)] = dists[s]
                row_dict['speaker{0}_doa'.format(s + 1)] = thetas[s]  # I changed from theta to doa
            for s in range(len(speakers_xy)):
                for dim_inx, dim in enumerate(dims):
                    if speakers_xy_np.ndim == 1:
                        row_dict['speaker{0}_{1}'.format(s + 1, dim)] = speakers_xy_np[dim_inx]
                    else:
                        row_dict['speaker{0}_{1}'.format(s + 1, dim)] = speakers_xy_np[s][dim_inx]
            writer.writerow(row_dict)
        file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='mchis')
    parser.add_argument('--num_results', type=int, default=1, metavar='NR',
                        help='Number of results')
    parser.add_argument('--mode', type=str, default='test',
                        help='Mode, i.e., train, val or test')
    parser.add_argument('--which_noise', type=str, default='with_wham_noise',  # 'with_bubble_noise',
                        help='Mode of noise, i.e., with_white_noise,with_bubble_noise,'
                            ' with_directional_white_noise or with_'
                            'diff_directioanl_white_noise')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('--mic_loc', type=list, default=[[22.85e-03, 22.85e-03, 0],
                                                        [-22.85e-03, 22.85e-03, 0],
                                                        [-22.85e-03, -22.85e-03, 0],
                                                        [22.85e-03, -22.85e-03, 0]],
                        # [[-82e-03, -29e-03, -5e-03], [1e-03, 30e-03, -1e-03],
                        # [77e-03, 11e-03, -2e-03], [83e-03, -60e-03, -5e-03],
                        # [-100e-03,-80e-03,-15e-03], [100e-03,-80e-03,-15e-03]],
                        help='microphone location on ARI')  # 'microphone location on the glasses')

    args = parser.parse_args()
    args_obj = vars(args)  # change the args object to a dictionary object
    stream = open("data_conifg_wham.yaml", 'r')
    yaml_dict = yaml.safe_load(stream)
    args_obj.update(yaml_dict)
    #fixed seed
    seed(args.seed)
    np.random.seed(args.seed)
    args.n_mics == len(args.mic_loc)
    Path(args.results_path).mkdir(exist_ok=True)
    # SAVE ARGS
    setting_path = os.path.join(args.results_path, args.mode, 'setting')
    Path(setting_path).mkdir(parents=True, exist_ok=True)
    file_name = os.path.join(setting_path, args.which_noise + '.txt')

    with open(file_name, 'wt') as opt_file:
        for k,v in sorted(
                args_obj.items()):  # items: make a dictionary as list of tuples that each tuple is a couple of (k,v)
            opt_file.write('%s: %s\n' % (str(k), str(v)))

    create_csv_results(args)
    print('done create data ;)')



