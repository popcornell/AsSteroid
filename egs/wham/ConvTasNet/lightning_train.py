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
        # inputs, targets, infos = batch
        # est_targets = self.model(inputs)
        # loss = self.loss_class.compute(targets, est_targets, infos=infos)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        tensorboard_logs = {'train_loss': loss}
        # return {'loss': loss.item()}
        # # REQUIRED
        # x, y = batch
        # y_hat = self.forward(x)
        # loss = F.cross_entropy(y_hat, y)
        # tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss = self.general_step(batch, batch_nb)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return self.optimizer

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self.loaders['train_loader']

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return self.loaders['val_loader']

    def on_save_checkpoint(self, checkpoint):
        # We can add anything to the dictionary, like, args and get_config.
        return checkpoint


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

    # First define the encoder and the decoder.
    # This can be either done by passing a string and the config
    # dictionary (with number of filters, filter size and stride, see conf.yml)
    # to fb.make_enc_dec.
    enc, dec = fb.make_enc_dec('free', **conf['filterbank'])
    # Or done by instantiating the filterbanks and passing them to the
    # Encoder and Decoder classes, as follows :
    # enc = fb.Encoder(fb.FreeFB(**conf['filterbank']))
    # dec = fb.Encoder(fb.FreeFB(**conf['filterbank']))

    # Define the mask network with input and output dimensions dictated by
    # by the encoder (also passing a dictionary defined in conf.yml).
    masker = TDConvNet(in_chan=enc.filterbank.n_feats_out,
                       out_chan=enc.filterbank.n_feats_out,
                       n_src=train_set.n_src, **conf['masknet'])
    # Pass the encoder, masker and decoder to the container class which
    # handles the forward for such architectures
    # model = nn.DataParallel(Container(enc, masker, dec))
    model = Container(enc, masker, dec)
    if conf['main_args']['use_cuda']:
        model.cuda()
    # Define Loss function
    loss_class = PITLossContainer(pairwise_neg_sisdr, n_src=train_set.n_src)
    # Define optimizer
    optimizer = make_optimizer(model.parameters(), **conf['optim'])

    # Pass everything to the solver with a training dicitonary defined in
    # the conf.yml file. Finally, call .train() and that's it.
    system = System(model, optimizer, loss_class, loaders)
    trainer = pl.Trainer(max_nb_epochs=2)
    trainer.fit(system)
    return trainer, system


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

    trainer, system = main(arg_dic)
