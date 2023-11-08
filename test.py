import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.pit_wrapper as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import matplotlib.pyplot as plt
import model.sdr as module_func_loss
import pandas as pd
import numpy as np
from model import combined_loss
from model.combined_loss import CombinedLoss
from model.combined_loss import reorder_source_mse
from model.combined_loss import calc_sisdr
from model.combined_loss import calc_sisdr_loss
from sklearn.metrics import confusion_matrix
from model.online_class_known_targets import OnlineSaving
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from Our_utils.utils_test import plot_spectrogram_masks, save_audio, save_vad, calc_vad
  


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    #print(config['tester']['csv_file_test'])
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['tester']['csv_file_test'],
        config['tester']['csd_lables'],
        batch_size=1,
        type_dataset=config['tester']['type_dataset'],
        shuffle=False,
        validation_split=0.0,
        num_workers=0
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get function handles of loss and metrics
    ##Separation loss
    func_loss_separation = getattr(module_func_loss, config["loss_separation"]["loss_func"])
    reduce = False
    if config["loss_separation"]["perm_reduce"] is not False:
        reduce = getattr(module_loss, config["loss_separation"]["perm_reduce"])
    else:
        reduce=None
    kw_separation = {"loss_func": func_loss_separation, "perm_reduce":reduce}
    criterion_separation = config.init_obj('loss_separation', module_loss, **kw_separation)

    ##VAD loss
    func_loss_vad = getattr(combined_loss, config["loss_vad"]["loss_func"])()
    kw_vad = {"loss_func": func_loss_vad}
    criterion_vad = config.init_obj('loss_vad', module_loss, **kw_vad)
    
    #THe combined loss
    criterion = CombinedLoss(criterion_separation, criterion_vad, **config["combined_loss"])

    metrics_separation = [getattr(module_metric, met) for met in config['metrics']["separation"]]
    metrics_vad = [getattr(module_metric, met) for met in config['metrics']["vad"]]
    metrics_vad_acc = [getattr(module_metric, met) for met in config['metrics']["vad_acc"]]
    metrics_separation_mix = [getattr(module_metric, met) for met in config['metrics']["separation_mix"]]
    metrics = {"separation": metrics_separation, "vad": metrics_vad, "vad_acc": metrics_vad_acc,
               "separation_mix": metrics_separation_mix}

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    save_test_path = config["tester"]["save_test"]
    print(f"The results will be saved in: {save_test_path}")
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=True) #strict=False

    model = model.to(device)
    model.eval()
    weights = config["combined_loss"]
    samplerate = 16e03
    do_calibaration = config["tester"]["calibaration"]["do_calibaration"]
    
    # Lists initialization
    reverbs, snrs, si_sdris, si_sdrs, si_sdrs_start, si_sdr_per_spk, si_sdr_per_spk_start, \
    vad_acc, vad_acc0, vad_acc1, predictions4calibaration, targets4calibaration = ([] for _ in range(14))
    # Dictionaries initialization
    value_separation, value_separation_start = {}, {}
    # Nested lists initialization
    tn, fp, fn, tp, tn_simple, fp_simple, fn_simple, tp_simple = ([[], []] for _ in range(8))

    online_bool = config['tester']['online']['online_bool']
    similarity = config['tester']['online']['similarity_bool']
    criterion_similarity = None
    if online_bool:
        if config["tester"]["online"]["l1_loss_similarity"]:
                similarity_func = torch.nn.L1Loss()
        elif config["tester"]["online"]["si_sdr_loss_similarity"]:
            similarity_func = calc_sisdr_loss
        if similarity:
            criterion_similarity = module_loss.PITLossWrapper(loss_func=similarity_func, pit_from="pw_pt")
        onlinesaving = OnlineSaving(criterion_separation, model, save_test_path, device, criterion_similarity)
            
    with torch.no_grad():
        for batch_idx, (sample_separation, label_csd) in enumerate(tqdm(data_loader)):
            reverb = sample_separation["reverb"]
            snr = sample_separation["snr"]
            data = sample_separation['mixed_signals']
            mix_without_noise, target_separation = sample_separation['mix_without_noise'], sample_separation['clean_speeches']
            data, target_separation, label_csd["vad_frames_individual"], label_csd["vad_frames_sum"], mix_without_noise = data.to(device), target_separation.to(device), \
                label_csd["vad_frames_individual"].to(device), label_csd["vad_frames_sum"].to(torch.long).to(device), mix_without_noise.to(device)

            if online_bool:
                #name_folder = f"batch_{batch_idx}_Reverb_{reverb.item():.2f}_Snr_{snr.item():.2f}"
                name_folder  = f"sample{batch_idx}/"
                onlinesaving.calc_online(data, target_separation, name_folder, batch_idx)
            
            out_separation, output_vad,  _ = model(data)
            if do_calibaration:
                predictions4calibaration.extend(output_vad.squeeze().tolist())
                targets4calibaration.extend(label_csd["vad_frames_individual"].squeeze().tolist())  
 
            
            ##Loss
            separation_loss, vad_indiv_loss, batch_indices_vad, batch_indices_separation, = \
                criterion(out_separation, target_separation, output_vad, label_csd["vad_frames_individual"])
            out_separation = reorder_source_mse(out_separation, batch_indices_separation)
            #reduce_kwargs = {'src': target} #I dont do reduce with csd
            
            si_sdr_spk = calc_sisdr(out_separation, target_separation) #shape=[B, num_spk]
            #print(si_sdr_spk.shape)
            si_sdr_per_spk.append(np.array(si_sdr_spk.detach().cpu())[0])
            
            ##the start si-sdr for each speaker
            mix = data.unsqueeze(dim=1)
            mix = mix.repeat(1, 2, 1)
            
            si_sdr_spk_start = calc_sisdr(mix, target_separation)
            si_sdr_per_spk_start.append(np.array(si_sdr_spk_start.detach().cpu())[0])
            
            masks = model.mask_per_speaker
            if config["arch"]["args"]["complex_masks"]:
                masks = torch.complex(real=masks[:, :, 0], imag=masks[:, :, 1]).abs()
            masks = reorder_source_mse(masks, batch_indices_separation)
            for i, metric in enumerate(metrics["separation"]):
                metric = metric.to(device)
                value_separation[metric.__name__] = metric(out_separation, target_separation)
                value_separation_start[metric.__name__] = metric(mix, target_separation).detach()
            if config["arch"]["args"]["final_vad"] and config["tester"]["type_dataset"] != "whamr":
                output_vad = reorder_source_mse(output_vad, batch_indices_vad)
                for i, metric in enumerate(metrics["vad"]):
                    metric = metric.to(device) 
                for i, metric in enumerate(metrics["vad_acc"]):
                    metric = metric.to(device)
                    vad_accuracy, acc0, acc1 = metric(output_vad, label_csd["vad_frames_individual"], batch_indices_vad) #change the output_vad to binary 
                    vad_acc0.append(acc0.item())
                    vad_acc1.append(acc1.item())
                    vad_acc.append(vad_accuracy.item())
                    for spk in range(output_vad.shape[1]):
                        tn_v, fp_v, fn_v, tp_v = confusion_matrix(label_csd["vad_frames_individual"][0, spk].cpu().numpy(), output_vad[0, spk].cpu().numpy(), normalize=None, labels=[0., 1.]).ravel()
                        tn[spk].append(tn_v)
                        fp[spk].append(fp_v)
                        fn[spk].append(fn_v)
                        tp[spk].append(tp_v)
            if config["tester"]["type_dataset"] != "whamr":            
                output_simple_vad = calc_vad(masks) #shape = [B, num_spk, T]
                for spk in range(output_simple_vad.shape[1]):
                    tn_v_simple, fp_v_simple, fn_v_simple, tp_v_simple = confusion_matrix(label_csd["vad_frames_individual"][0, spk].cpu().numpy(), output_simple_vad[0, spk].cpu().numpy(), normalize=None, labels=[0., 1.]).ravel()
                    tn_simple[spk].append(tn_v_simple)
                    fp_simple[spk].append(fp_v_simple)
                    fn_simple[spk].append(fn_v_simple)
                    tp_simple[spk].append(tp_v_simple)
            
            
            for i, metric in enumerate(metrics["separation_mix"]):
                metric = metric.to(device)
                si_sdri = metric(out_separation, target_separation, data)
            #print(si_sdri)
            
            mean_sisdr = torch.mean(si_sdr_spk).item()
            save_test_path_full = f"{save_test_path}Batch_{batch_idx}_SiSDRI_{si_sdri.item():.2f}_SiSDR_{mean_sisdr:.2f}_Reverb_{reverb.item():.2f}_Snr_{snr.item():.2f}_Stoi_{perceptual_value.item():.2f}/"

            if batch_idx < config["tester"]["num_save_samples"]: #save the first num_save_samples samples
                plot_spectrogram_masks(masks, 'Mask in fft domain (dB)', save_test_path_full)
                save_audio(data, out_separation, target_separation, save_test_path_full, samplerate, mean_sisdr)
                save_vad(output_vad, label_csd["vad_frames_individual"], save_test_path_full)  


            reverbs.append(reverb.item())
            snrs.append(snr.item())
            si_sdris.append(si_sdri.item())
            si_sdrs.append(value_separation["pit_si_sdr"].item())
            si_sdrs_start.append(value_separation_start["pit_si_sdr"].item())
 
    predictions4calibaration = torch.tensor(predictions4calibaration).cpu().numpy()
    targets4calibaration = torch.tensor(targets4calibaration).cpu().numpy()
    if do_calibaration:
        # Create a calibrated classifier using sklearn's CalibratedClassifierCV
        base_clf = GaussianNB()
        print("start the cali")
        #base_clf = SVC(gamma=2, C=1)#SVC(kernel="linear", C=0.025)#LinearSVC(random_state=0, tol=1e-5)
        print("build the LinearSVC")
        calibrated_classifier = CalibratedClassifierCV(base_estimator=base_clf, cv=2)
        print("going to fit the model")
        calibrated_classifier.fit(predictions4calibaration.reshape(-1, 1), targets4calibaration.ravel())
        print("fitted the model")
        prob_test = np.linspace(0, 1, num=10)
        print(f"the test points are: {prob_test}")

        cali_prob_test = calibrated_classifier.predict_proba(prob_test.reshape(-1, 1))
        print(f"the calibrated test points are: {cali_prob_test}")

        # Save the calibrated model
        joblib.dump(calibrated_classifier, save_test_path + 'calibrated_model_SVC.pkl')  
    
    si_sdr_per_spk = np.array(si_sdr_per_spk)
    si_sdr_per_spk_start = np.array(si_sdr_per_spk_start)
    
    if online_bool:
        online_sisdr = onlinesaving.online_sisdr
        reference_sisdr = onlinesaving.reference_sisdr

    n_samples = len(data_loader.sampler)
    scenario = np.arange(0, n_samples + 1) 
    data_list = [scenario, reverbs, snrs, si_sdris, si_sdrs, si_sdrs_start, 
            si_sdr_per_spk_start[:, 0], si_sdr_per_spk_start[:, 1], 
            si_sdr_per_spk[:, 0], si_sdr_per_spk[:, 1]]
    columns_list = ['scenario', 'reverb', 'snr', 'si_sdri', 'si_sdr', 'si_sdr_start', 
                    'si_sdr_start_speaker0', 'si_sdr_start_speaker1', 
                    'si_sdr_speaker0', 'si_sdr_speaker1']
    
    if weights["weight_vad_loss"]:
        data_list.extend([vad_acc, vad_acc0, vad_acc1, 
                        tp[0], fp[0], fn[0], tn[0], 
                        tp[1], fp[1], fn[1], tn[1], 
                        tn_simple[0], fp_simple[0], fn_simple[0], tp_simple[0], 
                        tn_simple[1], fp_simple[1], fn_simple[1], tp_simple[1]])

        columns_list.extend(['vad_accuracy', 'vad_acc_speaker0', 'vad_acc_speaker1', 
                            'tp_0', 'fp_0', 'fn_0', 'tn_0','tp_1', 'fp_1', 'fn_1', 'tn_1',
                            "tn_simple0", "fp_simple0", "fn_simple0", "tp_simple0", 
                            "tn_simple1", "fp_simple1", "fn_simple1"," tp_simple1"])

    df = pd.DataFrame(list(zip(*data_list)), columns=columns_list)    
    
    if online_bool:
        df["reference_sisdr"] = reference_sisdr
        df["online_sisdr"] = online_sisdr
        
    df.to_csv(save_test_path + 'results_information.csv')
    mean_sdr = np.mean(si_sdrs)      
    print(f"the mean si_sdr is: {mean_sdr}")
    fig, axs = plt.subplots(1, 1)
    axs.hist(si_sdrs, bins="sqrt", density=True)
    axs.set_title(f"The mean sdr is {mean_sdr}")
    plt.savefig(f"{save_test_path}hist_si_sdr.png")
    
    if online_bool:
        logger.info(f"The online_sisdr is: {np.mean(online_sisdr)},\t The reference_sisdr is: {np.mean(reference_sisdr)}")
        mean_sdr_online = np.mean(online_sisdr) 
        print(f"the mean si_sdr is: {mean_sdr_online}")
        fig, axs = plt.subplots(1, 1)
        axs.hist(online_sisdr, bins="sqrt", density=True)
        axs.set_title(f"The mean sdr is {mean_sdr_online}")
        plt.savefig(f"{save_test_path}hist_si_sdr_online.png")
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="/home/dsi/moradim/Audio-Visual-separation-using-RTF/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="/dsi/gannot-lab1/datasets/mordehay/Result/Separation_data_wham_partial_overlap_ModoluDial_Vad099_decay095_inputActivity_AttentionBeforeSum_2rSkippLN/models/AV_model/0108_114521/model_best.pth",
                                    type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args, trainer_or_tester="tester", save_path="save_test")
    main(config)
