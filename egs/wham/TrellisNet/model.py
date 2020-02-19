from torch import nn

import asteroid.filterbanks as fb
from trellisnet import TrellisNet
from asteroid.engine.optimizers import make_optimizer
from asteroid.masknn import norms
from asteroid.masknn import activations
from asteroid.utils import has_arg
import torch

class Net(nn.Module):

    def __init__(self, in_chan, n_src, out_chan=None, bn_chan=64, hid_chan=256, nlevels=40, ksz=2,
                 dropout=0.0, mask_act="sigmoid"):
        super(Net, self).__init__()

        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.hid_chan = hid_chan
        self.n_src = n_src
        layer_norm = norms.get('cLN')(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        self.net = TrellisNet(bn_chan,hid_chan-out_chan, out_chan, nlevels, ksz, dropout, aux_frequency=-1)

        mask_conv = nn.Conv1d(out_chan, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, x):

        x = self.bottleneck(x)
        B, C, T = x.size()
        dummy = torch.zeros((2, self.hid_chan, 1))
        out = self.net(x, dummy, False)[0]
        out = out.transpose(1, -1)

        score = self.mask_net(out)
        score = score.view(B, self.n_src, self.out_chan, T)
        est_mask = self.output_act(score)
        return est_mask

class Model(nn.Module):
    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        est_masks = self.masker(tf_rep)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        return self.pad_output_to_inp(self.decoder(masked_tf_rep), x)

    @staticmethod
    def pad_output_to_inp(output, inp):
        """ Pad first argument to have same size as second argument"""
        inp_len = inp.size(-1)
        output_len = output.size(-1)
        return nn.functional.pad(output, [0, inp_len - output_len])


def make_model_and_optimizer(conf):
    """ Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    # Define building blocks for local model
    enc, dec = fb.make_enc_dec('free', **conf['filterbank'])
    masker = Net(in_chan=enc.filterbank.n_feats_out,
                       out_chan=enc.filterbank.n_feats_out, n_src=2
                       )
    model = Model(enc, masker, dec)
    # Define optimizer of this model
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    return model, optimizer
