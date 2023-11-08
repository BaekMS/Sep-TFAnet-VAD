from email.policy import strict
from pickletools import optimize
import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import wandb
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, lr_scheduler):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        
        cfg_trainer = config['trainer']
        self.accum = cfg_trainer["accum_iter"]
        self.max_clip = cfg_trainer['max_clip']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.vad_decay_weight = cfg_trainer['vad_decay_weight']
        self.mse_decay_weight = cfg_trainer['mse_decay_weight']
        self.com_sisdr_decay_weight = cfg_trainer['com_sisdr_decay_weight']
        self.cs_decay_weight = cfg_trainer['cs_decay_weight']
        ###############
        #self.swa_model = AveragedModel(model)
        self.swa_start = cfg_trainer['swa_start']
        self.swa_scheduler = SWALR(optimizer, swa_lr=0.05, anneal_epochs=2)##lr was succsed with lr=0.05
        self.lr_scheduler = lr_scheduler
        ################

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        #self.loss_mean_list = 0
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        idx = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            """ Only if we toggle the model
            if idx <= 9:
                print("The CSD is training")
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.csd_model.parameters():
                    param.requires_grad = True

            if idx > 9 and idx <= 19:
                if idx == 19:
                    idx = 0
                print("The Separation + CSD is training")
                for param in self.model.parameters():
                    param.requires_grad = True
                for param in self.model.csd_model.parameters():
                    param.requires_grad = False"""
            
            """for param in self.model.parameters():
                param.requires_grad = False  """      
            #num_learnable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            #print(f"There is {num_learnable} learnable weights at this phase")
            result = self._train_epoch(epoch)
            #self.logger.info(f"The mean train loss is {self.loss_mean_list}")

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            wandb.log(log)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
                #wandb.log({str(key): value})

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                self.logger.info(improved)
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                self.logger.info(not_improved_count)
                
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            if best:
                self._save_checkpoint(epoch, save_best=best)
            """if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)"""


    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if epoch > self.swa_start:
            self.logger.info("swa model")
            model = self.swa_model
        else:
            self.logger.info("regular model")
            model = self.model
        arch = type(model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        """filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))"""
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = 1#checkpoint['epoch'] + 1
        self.mnt_best = -1000#checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))