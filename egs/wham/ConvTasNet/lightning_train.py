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

import pytorch_lightning as pl

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--model_path', default='exp/tmp/final.pth',
                    help='Full path to save best validation model')


# ============================LIGHTNING Specific =============================
""" General comments + things on the way 
The organization of the pl.LightningModule children in the repo is based on
Namespace, all parameters are directly given from arparse. We don't want 
 that here because it's not pretty either transparent.
 We can make a module containing model, optimizer, loss_class loaders 
 and some system specific config.
 
 On Trainer, for now it has 30 arguments, most of them are good with 
 defaults values. But the design is a bit worrying IMO
 It can also be subclassed to have only the part we are interested in, but 
 that can come later
 """


class System(pl.LightningModule):
    def __init__(self, model, optimizer, loss_class, loaders):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_class = loss_class
        self.loaders = loaders
        self.hparams = argparse.Namespace(useless=True)
        # self.hparams = self.model.get_config()

    def general_step(self, batch, batch_nb):
        inputs, targets, infos = batch
        est_targets = self.model(inputs)
        loss = self.loss_class.compute(targets, est_targets, infos=infos)
        return loss

    def training_step(self, batch, batch_nb):
        loss = self.general_step(batch, batch_nb)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss = self.general_step(batch, batch_nb)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    """ Maybe it can be subclassed to automatize the asignment of optimizer 
    and loaders because we feed it in the constructor."""
    def configure_optimizers(self):
        return self.optimizer

    @pl.data_loader
    def train_dataloader(self):
        return self.loaders['train_loader']

    @pl.data_loader
    def val_dataloader(self):
        return self.loaders['val_loader']

    """ That's the method we can change to add any key we want to the 
    dictionary we will save. """
    def on_save_checkpoint(self, checkpoint):
        return checkpoint

    # ========================================================================


def main(conf):
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
    enc, dec = fb.make_enc_dec('free', **conf['filterbank'])
    masker = TDConvNet(in_chan=enc.filterbank.n_feats_out,
                       out_chan=enc.filterbank.n_feats_out,
                       n_src=train_set.n_src, **conf['masknet'])

    model = Container(enc, masker, dec)
    if conf['main_args']['use_cuda']:
        model.cuda()
    loss_class = PITLossContainer(pairwise_neg_sisdr, n_src=train_set.n_src)
    # Define optimizer
    optimizer = make_optimizer(model.parameters(), **conf['optim'])


    # ============================LIGHTNING Specific =========================
    system = System(model, optimizer, loss_class, loaders)
    trainer = pl.Trainer(max_nb_epochs=2)
    trainer.fit(system)
    # ========================================================================


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
