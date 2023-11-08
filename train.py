import argparse
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.pit_wrapper as module_loss
import model.sdr as module_func_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import wandb
from model.combined_loss import CombinedLoss
from model import combined_loss
import os 

##data:
#"/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/train/csv_files/with_wham_noise_res.csv"
# "/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/train/csv_files/with_wham_noise_res_more_informative.csv"
# "/dsi/gannot-lab/datasets/mordehay/data_wham_partial_wsj0_sir0_fixed/train/csv_files/with_wham_noise_res.csv"

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#aa
def main(config):
    
    wandb.init(project="Separation_With_Vad", entity="ren_mor")
    wandb.run.name = os.path.basename(config["trainer"]["save_dir"])
    wandb.save("config.json", policy="now")
    wandb.save("model/model.py", policy="now")
    wandb.save("trainer/trainer.py", policy="now")
    wandb.save("model/combined_loss.py", policy="now")
    logger = config.get_logger('train')
    print(config["trainer"]["save_dir"])
    print(torch.cuda.is_available())
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)     
    #logger.info(model)
    logger.info(f"The seed is {SEED}")

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    #device = 'cpu'
    print(f"the device is : {device}")
    model = model.to(device)       
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    func_loss_separation = getattr(module_func_loss, config["loss_separation"]["loss_func"]).to(device)
    reduce = None
    kw_separation = {"loss_func": func_loss_separation, "perm_reduce":reduce}
    criterion_separation = config.init_obj('loss_separation', module_loss, **kw_separation)
    func_loss_vad = getattr(combined_loss, config["loss_vad"]["loss_func"])()
    kw_vad = {"loss_func": func_loss_vad}
    criterion_vad = config.init_obj('loss_vad', module_loss, **kw_vad)
    criterion = CombinedLoss(criterion_separation, criterion_vad, config["combined_loss"])

    #metrics
    metrics_vad_acc = [getattr(module_metric, met) for met in config['metrics']["vad_acc"]]
    metrics_separation = [getattr(module_metric, met) for met in config['metrics']["separation"]]
    metrics_vad = [getattr(module_metric, met) for met in config['metrics']["vad"]]
    metrics_separation_mix = [getattr(module_metric, met) for met in config['metrics']["separation_mix"]]
    metrics = {"separation": metrics_separation, "vad": metrics_vad, "vad_acc": metrics_vad_acc,
               "separation_mix": metrics_separation_mix}


    wandb.watch(model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_of_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of parameters that require grad in the model is: {num}".format(num=num_of_param))
    optimizer = config.init_obj('optimizer', torch.optim, list(trainable_params))

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    
    
    logger.info(f"The save dir is {config['trainer']['save_dir']}")
    trainer.train()


if __name__ == '__main__':
    print("Begin!")
    args = argparse.ArgumentParser(description='Separation With Vad')
    args.add_argument('-c', '--config', default="config_with_vad.json",
                      type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    config = ConfigParser.from_args(args, trainer_or_tester="trainer",  save_path="save_dir")
    main(config)

