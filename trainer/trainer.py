import numpy as np
from numpy.testing import print_assert_equal
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from torch import autograd
import torch.nn as nn
from model.combined_loss import reorder_source_mse



class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, lr_scheduler)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.weights = self.criterion.weights
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns["separation"]],
            *[m.__name__ for m in self.metric_ftns["vad"]], *[m.__name__ for m in self.metric_ftns["vad_acc"]],\
            *[m.__name__ for m in self.metric_ftns["separation_mix"]], 'separation_loss')
        
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns["separation"]],
            *[m.__name__ for m in self.metric_ftns["vad"]], *[m.__name__ for m in self.metric_ftns["vad_acc"]],\
            *[m.__name__ for m in self.metric_ftns["separation_mix"]], 'separation_loss')
                                           
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (sample_separation, label_csd) in enumerate(self.data_loader):
            data = sample_separation['mixed_signals']
            target_separation = sample_separation['clean_speeches']
            data, target_separation, label_csd["vad_frames_individual"] =\
                data.to(self.device), target_separation.to(self.device), label_csd["vad_frames_individual"].to(self.device)
            out_separation, output_vad, _  = self.model(data)
            separation_loss, vad_indiv_loss, batch_indices_vad, batch_indices_separation = \
                self.criterion(out_separation, target_separation, output_vad, \
                               label_csd["vad_frames_individual"])
            loss = separation_loss * self.weights["weight_separation_loss"] + vad_indiv_loss * self.weights["weight_vad_loss"]
            loss = loss / self.accum
            loss.backward()

            if ((batch_idx + 1) % self.accum == 0) or (batch_idx + 1 == len(self.data_loader)):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_clip, norm_type=2)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.train_metrics.update('loss', self.accum *loss.item())
                for met in self.metric_ftns["separation"]:
                    met = met.to(self.device)
                    self.train_metrics.update(met.__name__, met(out_separation, target_separation).item())
                if self.weights["weight_vad_loss"]:
                    output_vad = reorder_source_mse(output_vad, batch_indices_vad)
                    for met in self.metric_ftns["vad"]:
                        met = met.to(self.device)
                        self.train_metrics.update(met.__name__, met(output_vad, label_csd["vad_frames_individual"]).item()) 
                    for met in self.metric_ftns["vad_acc"]:
                        met = met.to(self.device)
                        self.train_metrics.update(met.__name__, met(output_vad, label_csd["vad_frames_individual"],
                                                                    batch_indices_vad)[0].item())
                for met in self.metric_ftns["separation_mix"]:
                    met = met.to(self.device)
                    self.train_metrics.update(met.__name__, met(out_separation, target_separation, data).item())
                self.train_metrics.update('separation_loss', separation_loss.item())
                
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Total_Loss: {:.6f}, Separation_Loss: {:.6f}, Vad_Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    self.accum * loss.item(),
                    separation_loss.item(),
                    vad_indiv_loss.item()))
                if self.criterion.learn_weight_bool:
                    self.logger.info(f"The learn_weight_vadLoss is: {self.criterion.learn_weight_vadLoss}")
                    self.logger.info(f"The learn_weight_separationLoss is: {self.criterion.learn_weight_separationLoss}")
                if epoch <= self.swa_start:
                    if self.config["lr_scheduler"]["type"] == "CosineAnnealingWarmRestarts":
                        self.lr_scheduler.step(epoch-1) #for cosine-anneling scheduler
                        self.logger.info(f"The lr is: {self.lr_scheduler.get_last_lr()[0]:.05f}")#
            
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, _ = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if epoch > self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                self.logger.debug(f"The swa lr is: {self.swa_scheduler.get_last_lr()[0]:.03f}")
        else:
            if self.config["lr_scheduler"]["type"] == "CosineAnnealingWarmRestarts":
                pass
            elif  self.config["lr_scheduler"]["type"] == "ReduceLROnPlateau":
                self.lr_scheduler.step(log["val_loss"]) #it's for OnPlatuo schduler
                self.logger.info(f"The lr is: {self.lr_scheduler._last_lr[0]:.05f}")
                
        self.weights["weight_vad_loss"] = self.weights["weight_vad_loss"] * self.vad_decay_weight
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if epoch > self.swa_start:
            self.logger.info("swa model")
            model = self.swa_model
        else:
            self.logger.info("regular model")
            model = self.model
        model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (sample_separation, label_csd) in enumerate(self.valid_data_loader):
                data = sample_separation['mixed_signals']
                target_separation = sample_separation['clean_speeches']
                data, target_separation, label_csd["vad_frames_individual"] = \
                    data.to(self.device), target_separation.to(self.device), \
                    label_csd["vad_frames_individual"].to(self.device)
                out_separation, output_vad,  _ = model(data)
                separation_loss, vad_indiv_loss, batch_indices_vad, batch_indices_separation = \
                    self.criterion(out_separation, target_separation, output_vad, label_csd["vad_frames_individual"])
                loss = separation_loss * self.weights["weight_separation_loss"] + vad_indiv_loss * self.weights["weight_vad_loss"]
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns["separation"]:
                    met = met.to(self.device)
                    self.valid_metrics.update(met.__name__, met(out_separation, target_separation).item())
                
                if self.weights["weight_vad_loss"]:
                    output_vad = reorder_source_mse(output_vad, batch_indices_vad)
                    for met in self.metric_ftns["vad"]:
                        met = met.to(self.device)
                        self.valid_metrics.update(met.__name__, met(output_vad, label_csd["vad_frames_individual"]).item()) 
                    for met in self.metric_ftns["vad_acc"]:
                        met = met.to(self.device)
                        self.valid_metrics.update(met.__name__, met(output_vad, label_csd["vad_frames_individual"],
                                                                    batch_indices_vad)[0].item())
                for met in self.metric_ftns["separation_mix"]:
                    met = met.to(self.device)
                    self.valid_metrics.update(met.__name__, met(out_separation, target_separation, data).item())
                self.valid_metrics.update('separation_loss',separation_loss.item())
                
        return self.valid_metrics.result(), loss

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)