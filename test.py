import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as data_module
import model.loss as loss_module
import model.metric as metric_module
import model.model as arch_module
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup dataloader instances
    # TODO: update data loader
    data_loader = getattr(data_module, config['data_loader']['type'])(
        csv_file=config['data_loader']['args']['csv_file'],
        data_dir=config['data_loader']['args']['data_dir'],
        batch_size=256,
        shuffle=False,
        num_workers=2,
        training=False
    )

    # build model architecture, print to console
    model = config.init_obj('arch', arch_module)
    logger.info(model)

    # get function handles for loss and metrics
    criterion = getattr(loss_module, config['loss'])
    metrics = [getattr(metric_module, metric) for metric in config['metrics']]

    # load checkpoint
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            # put on device
            data, target = data.to(device), target.to(device)
            output = model(data)

            # ====================================================
            # save sample images, or do something with output here
            # ====================================================

            # compute loss, metrics on test dataset
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    # store average loss
    log = {'loss': total_loss / n_samples}
    log.update({
    met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Hair-CAM PyTorch Project')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
