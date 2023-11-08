import numpy as np  
import argparse
import torch
import model.pit_wrapper as module_loss
import model.model as module_arch
from parse_config import ConfigParser
import numpy as np
from scipy.io.wavfile import read
import torchaudio.transforms as T
from model.online_class_unknown_targets import OnlineSaving
from Our_utils.utlis_inference import plot_spectrogram, save_audio, save_vad
import json
import copy



def parse_dictionary(dictionary_string):
    """Convert a string in dictionary format to a dictionary object."""
    print(dictionary_string)
    try:
        return json.loads(dictionary_string)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary provided: {dictionary_string}. Error: {e}")


        
def main(config):
    logger = config.get_logger('test_real')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    #logger.info(model)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # Load the state dictionary from the .pth file
    
    
    """checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    
    # List of keys to remove
    keys_to_remove = ["to_spec.weight", "to_spec.bias"]

    # Remove the keys from the state dictionary
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]

    # Save the modified state dictionary back to the .pth file
    checkpoint_new = {
            'arch': checkpoint["arch"],
            'epoch': 315,
            'state_dict': state_dict,
            'optimizer': checkpoint["arch"],
            'monitor_best': checkpoint["monitor_best"]
        }
    torch.save(checkpoint_new, 'model_without_vad_without_config.pth')"""
    checkpoint = torch.load(config.resume, map_location="cpu")
    state_dict = checkpoint['state_dict']
    save_test_path = config.args.save_test_path
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    online_bool = config.args.online
    path_audio = config.args.path_mix
    inference_kw = copy.deepcopy(default_inference_kw)
    inference_kw_user = config.args.inference_kw
    inference_kw.update(inference_kw_user)
    
    samplerate, audio = read(path_audio)
    audio = np.array(audio, dtype=np.float32)
    if audio.ndim > 1:
        print("The audio is not mono, the first channel was chosen")
        if audio.shape[1] > audio.shape[0]:
            audio = audio[0]
        else:
            audio = audio[:, 0]
    if samplerate != 16000:
        print("The audio is not 16KHz, resmapling to 16KHz..")
        resample = T.Resample(samplerate, 16000, dtype=torch.float32)
        audio = resample(torch.tensor(audio))
            
    normalized_audio = 1.8*(audio - audio.min()) / (audio.max() - audio.min()) - 0.9
    normalized_audio = torch.from_numpy(normalized_audio)
    normalized_audio = torch.unsqueeze(normalized_audio, dim=0) #make the shape [1, num_of_samples]
    if online_bool:
        similarity_func = torch.nn.L1Loss()
        criterion_similarity = module_loss.PITLossWrapper(loss_func=similarity_func, pit_from="pw_pt")
        onlinesaving = OnlineSaving(model, save_test_path, criterion_similarity)
        name_folder = "online_results"
        onlinesaving.calc_online(normalized_audio, name_folder, 0, inference_kw)
    with torch.no_grad():
        out_separation, output_vad,  _ = model(normalized_audio, inference_kw) #the output's shape is [1, 2, num_of_samples]
    masks = model.mask_per_speaker
    
    plot_spectrogram(masks, 'Mask in fft domain', save_test_path)
    save_audio(normalized_audio, out_separation, save_test_path, config.args.precision_save)
    if config.resume == "model_without_vad.pth":
        save_vad(output_vad, save_test_path)


if __name__ == '__main__':
    
    default_inference_kw = {
    "filter_signals_by_smo_vad": False,
    "filter_signals_by_unsmo_vad": False,
    "length_smoothing_filter": 3 ,
    "threshold_activated_vad": 0.5,
    "return_smoothed_vad": False
    }
    
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config_without_vad.json", type=str,
                      help='config file path (default: None)')
    #"/dsi/scratch/from_netapp/users/mordehay/Results/Separation_data_wham_partial_overlap_ModoluDial_inputActivity_AttentionBeforeSum_ResidualLN_resume/models/AV_model/0115_172306/model_best.pth"
    # /home/dsi/moradim/separation-with-vad/model_with_vad_without_config.pth
    "/dsi/gannot-lab1/datasets/mordehay/Result/Separation_data_wham_partial_overlap_ModoluDial_Vad099_decay095_inputActivity_AttentionBeforeSum_2rSkippLN/models/AV_model/0108_114521/model_best.pth"
    args.add_argument('-r', '--resume', default="model_without_vad.pth",
                                    type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="1", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-sp', '--save_test_path', default="results_withoutvad", type=str,
                      help='where to save the separated signals (default: current dir))')
    args.add_argument('-o', '--online', default=True, type=bool,
                      help='whether to run the online mode or not (default: False)')
    args.add_argument('-ps', '--precision_save', default=32, choices=[16, 32], type=int,
                      help='The precision of the separated signals (default: 32)')
    args.add_argument('-pm', '--path_mix', type=str, required=True,
                      help='The path to the mix signal on which we want to run the separation')
    args.add_argument('-ikw', '--inference_kw', type=parse_dictionary, default={},
                      help='Here you can type the options for the inference phase.'
                        'filter_signals_by_smo_vad - whether to filter the separated signals by the smoothed VAD or not (default: False)'
                        'filter_signals_by_unsmo_vad - whether to filter the separated signals by the unsmoothed VAD or not (default: False)'
                        'length_smoothing_filter - the length of the smoothing filter (default: 3)'
                        'threshold_activated_vad - the threshold for the VAD (default: 0.5)'
                        'return_smoothed_vad - whether to return the smoothed VAD or not (default: False)')
    config = ConfigParser.from_args(args, trainer_or_tester="tester", save_path="save_test_real")
    main(config)

