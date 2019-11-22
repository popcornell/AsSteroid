import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn

from asteroid import Container, Solver
import asteroid.filterbanks as fb
from asteroid.masknn import TDConvNet
from asteroid.engine.losses import PITLossContainer, pairwise_neg_sisdr
from asteroid.data.wham_dataset import WhamDataset
from asteroid.engine.optimizers import make_optimizer

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--model_path', default='exp/tmp/final.pth',
                    help='Full path to save best validation model')


def main(conf):
    # Define data pipeline with datasets and loaders
    train_set = WhamDataset(conf['data']['train_dir'], conf['data']['task'],
                            sample_rate=conf['data']['sample_rate'],
                            nondefault_nsrc=conf['data']['nondefault_nsrc'])
    val_set = WhamDataset(conf['data']['valid_dir'], conf['data']['task'],
                          sample_rate=conf['data']['sample_rate'],
                          nondefault_nsrc=conf['data']['nondefault_nsrc'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['data']['batch_size'],
                              num_workers=conf['data']['num_workers'])
    val_loader = DataLoader(val_set, shuffle=True,
                            batch_size=conf['data']['batch_size'],
                            num_workers=conf['data']['num_workers'])
    loaders = {'train_loader': train_loader, 'val_loader': val_loader}

    # Define model
    enc, dec = fb.make_enc_dec('free', **conf['filterbank'])
    masker = TDConvNet(in_chan=enc.filterbank.n_feats_out,
                       out_chan=enc.filterbank.n_feats_out,
                       n_src=train_set.n_src, **conf['masknet'])
    model = nn.DataParallel(Container(enc, masker, dec))
    if conf['main_args']['use_cuda']:
        model.cuda()
    loss_class = PITLossContainer(pairwise_neg_sisdr, n_src=train_set.n_src)
    optimizer = make_optimizer(model.parameters(), **conf['optim'])

    """ Comments
    The function `step` uses global variables and that's how ignite was 
    designed. I'm not sure it cannot have some bad behaviour but it's not so
    pretty.
    It needs quite a lot of lines to define checkpointing and callbacks but it 
    looks easily extendable (also with lots of lines).
    """

    # ============================== IGNITE SPECIFIC ==========================
    def step(engine, data):
        inputs, targets, infos = data
        est_targets = model(inputs)
        loss = loss_class.compute(targets, est_targets, infos=infos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}

    trainer = Engine(step)
    output_dir = 'exp/tmp/'
    CKPT_PREFIX = 'model.ckpt'
    checkpoint_handler = ModelCheckpoint(output_dir, CKPT_PREFIX,
                                         save_interval=1, n_saved=10,
                                         require_empty=False)
    # adding handlers using `trainer.add_event_handler` method API
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                              handler=checkpoint_handler,
                              to_save={'model': model})
    # # attach running average metrics
    monitoring_metrics = ['loss']
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')

    # attach progress bar
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)
    timer = Timer(average=True)
    trainer.run(loaders['train_loader'], 2)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message('Epoch {} done. Time per batch: '
                         '{:.3f}[s]'.format(engine.state.epoch, timer.value()))
        timer.reset()

    # =========================================================================


if __name__ == '__main__':
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to falicitate calls to the different
    # asteroid objects (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)

    main(arg_dic)
