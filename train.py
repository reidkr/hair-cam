import argparse
import collections
import numpy as np
import torch
import data_loader.data_loaders as data_module
import model.loss as loss_module
import model.metric as metric_module
import model.model as arch_module
from parse_config import ConfigParser
from trainer import Trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup dataloader instances
    train_loader, valid_loader = config.init_obj('data_loader', data_module)

    # build model architecture, print to console
    model = config.init_obj('arch', arch_module)
    logger.info(model)

    # get function handles for loss and metrics

    criterion = getattr(loss_module, config['loss'])
    metrics = [getattr(metric_module, metric) for metric in config['metrics']]

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # loss function weights
    # weights = torch.FloatTensor(np.array([]))

    # setup trainer
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer,
                      metric_fcns=metrics, config=config, data_loader=train_loader,
                      valid_data_loader=valid_loader, lr_scheduler=lr_scheduler,
                      weights=None)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Hair-CAM PyTorch Project')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom CLI options to modify config from default values in json config file
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
    CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
