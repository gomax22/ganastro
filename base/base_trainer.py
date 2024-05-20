# https://github.com/victoresque/pytorch-template/blob/master/base/base_trainer.py

import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, models, criterion, metric_ftns, optimizers, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.models = models
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizers = optimizers

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitors = cfg_trainer.get('monitor', ['off', 'off'])

        # configuration to monitor model performance and save best
        self.mnt_best = [None, None]
        self.mnt_mode, self.mnt_metric = [None, None], [None, None]
        self.early_stops = cfg_trainer.get('early_stop', [inf, inf])
        
        for idx in range(2):
            if self.monitors[idx] == 'off':
                self.mnt_mode[idx] = 'off'
            else:
                self.mnt_mode[idx], self.mnt_metric[idx] = self.monitors[idx].split()
                assert self.mnt_mode[idx] in ['min', 'max']
                self.mnt_best[idx] = inf if self.mnt_mode == 'min' else -inf
                
                if self.early_stops[idx] <= 0:
                    self.early_stops[idx] = inf
                    
        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

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
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs + 1):
            result = self._train_epoch(epoch)
            # save logged informations into log dict
            log = {'epoch': epoch}
            for res in result:
                log.update(res)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            for idx in range(2):
                best = False
                if self.mnt_mode[idx] != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode[idx] == 'min' and log[self.mnt_metric[idx]] <= self.mnt_best[idx]) or \
                                (self.mnt_mode[idx] == 'max' and log[self.mnt_metric[idx]] >= self.mnt_best[idx])
                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode[idx] = 'off'
                        improved = False

                    if improved:
                        self.mnt_best[idx] = log[self.mnt_metric[idx]]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                        "Training stops.".format(self.early_stop))
                        break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        generator_state = {
            'type': type(self.models[0].module).__name__ if type(self.models[0]).__name__ == 'DataParallel' else type(self.models[0]).__name__ , 
            'state_dict': self.models[0].state_dict(),
            'optimizer': self.optimizers[0].state_dict(),
            'monitor_best': self.mnt_best[0],
            'config': self.config
        }
        
        discriminator_state = {
            'type': type(self.models[1].module).__name__ if type(self.models[1]).__name__ == 'DataParallel' else type(self.models[1]).__name__, 
            'state_dict': self.models[1].state_dict(),
            'optimizer': self.optimizers[1].state_dict(),
            'monitor_best': self.mnt_best[1],
            'config': self.config
        }
        
        state = {
            'epoch': epoch,
            'generator': generator_state,
            'discriminator': discriminator_state
        }
        
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
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
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = [checkpoint['generator']['monitor_best'], checkpoint['discriminator']['monitor_best']]

        # load architecture params from checkpoint.
        if checkpoint['generator']['type'] != self.config['generator']['type']:
            self.logger.warning("Warning: Architecture configuration (Generator) given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.models[0].load_state_dict(checkpoint['generator']['state_dict'])
        
        # load architecture params from checkpoint.
        if checkpoint['discriminator']['type'] != self.config['discriminator']['type']:
            self.logger.warning("Warning: Architecture configuration (Discriminator) given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.models[1].load_state_dict(checkpoint['discriminator']['state_dict'])
        

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['generator']['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type (Generator) given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizers[0].load_state_dict(checkpoint['generator']['optimizer'])
            
            
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['discriminator']['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type (Discriminator) given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizers[1].load_state_dict(checkpoint['discriminator']['optimizer'])
            
        
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))