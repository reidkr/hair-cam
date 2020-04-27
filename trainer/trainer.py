import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, optimizer, metric_fcns, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None,
                 weights=None, len_epoch=None):
        # call parent's (in this case BaseTrainer) ``__init__`` method,
        # setting all it's attributes
        super().__init__(model, criterion, optimizer, metric_fcns, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # do epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # do iteration-based training:
            # wrapper for endless dataloader
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.weights = weights
        # number of batches at which to write to log
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # instantiate train and valid metric tracker
        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_fcns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_fcns], writer=self.writer
        )

    def _train_epoch(self, epoch):
        '''
        Training logic for an epoch
        :param epoch: Integer, current training epoch
        :return: Log containing average loss and metric(s) for this epoch
        '''
        # set model to training mode
        self.model.train()
        # reset training metrics
        self.train_metrics.reset()
        # iterate over sample batches
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # move data to device 'cuda:0' if GPU available, else CPU
            data, target = data.to(self.device), target.to(self.device)

            # reset gradients
            self.optimizer.zero_grad()
            # do forward pass
            output = self.model(data)
            # get loss
            loss = self.criterion(output, target, weights=self.weights)
            # compute gradients
            loss.backward()
            # update model weights/params
            self.optimizer.step()

            # update training metrics
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())  # update training loss
            for metric in self.metric_fcns:  # update training metrics
                self.train_metrics.update(metric.__name__, metric(output, target))

            # write to log if iterated over `log_step` batches
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    'Train epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))

            if batch_idx == self.len_epoch:
                break

            # log info on train and valid metrics
            log = self.train_metrics.result()
            if self.do_validation:
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_' + key: val for key, val in val_log.items()})

            if self.lr_scheduler is not None:
                # update learning rate
                self.lr_scheduler.step()

            return log

    def _valid_epoch(self, epoch):
        '''
        Validation after training an epoch

        :param epoch: Integer, current training epoch
        :return: log containing info about validation
        '''
        # set model to eval mode
        self.model.eval()
        # reset validation metrics
        self.valid_metrics.reset()
        # don't need to track history for validation, context reduces memory usage
        with torch.no_grad():
            # iterate over sample batches in validation set
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                # move data to device, 'cuda:0' if GPU available, else CPU
                data, target = data.to(self.device), target.to(self.device)

                # do forward pass
                output = self.model(data)
                # get loss
                loss = self.criterion(output, target, weights=self.weights)

                # update validation metrics
                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for metric in self.metric_fcns:
                    self.valid_metrics.update(metric.__name__, metric(output, target))
                self.writer.add_image(
                    'input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to tensorboard
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        '''
        TODO: comment
        :param batch_idx: Integer, current batch number
        '''
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
