from torch import nn
import torch
from asteroid.masknn import norms
from asteroid.masknn import activations
from asteroid.utils import has_arg
from asteroid.masknn.blocks import Conv1DBlock


class SepConv1DBlock(nn.Module):

    def __init__(self, in_chan, in_chan_spk_vec, hid_chan, skip_out_chan, kernel_size, padding,
                 dilation, norm_type="gLN"):
        super(SepConv1DBlock, self).__init__()
        conv_norm = norms.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        self.depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)
        self.squeeze = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan))

        self.unsqueeze = nn.Sequential(nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

        # FiLM conditioning
        self.mul_lin = nn.Linear(in_chan_spk_vec, hid_chan)
        self.add_lin = nn.Linear(in_chan_spk_vec, hid_chan)

    def apply_conditioning(self, spk_vec, squeezed):
        B, embed, frames = spk_vec.size()
        hid_size = squeezed.size(1)
        spk_vec = spk_vec.reshape(B, embed, frames)

        spk_vec = spk_vec.transpose(1, -1).reshape(B*frames, embed)
        multiply = self.mul_lin(spk_vec).reshape(B, frames, hid_size ).transpose(1, -1)
        bias = self.add_lin(spk_vec).reshape(B, frames, hid_size).transpose(1, -1)

        return multiply*squeezed + bias


    def forward(self, x, spk_vec):
        """ Input shape [batch, feats, seq]"""
        squeezed = self.squeeze(x)

        conditioned = self.apply_conditioning(spk_vec, self.depth_conv1d(squeezed))
        unsqueezed = self.unsqueeze(conditioned)
        res_out = self.res_conv(unsqueezed)
        skip_out = self.skip_conv(unsqueezed)
        return res_out, skip_out

class SpeakerStack(nn.Module):

    # basically this is plain conv-tasnet remove this in future releases

    def __init__(self, in_chan, n_src, out_chan=32, n_blocks=8, n_repeats=3,
                 bn_chan=128, hid_chan=512, skip_chan=128, kernel_size=3,
                 norm_type="gLN", mask_act="linear"):
        
        super(SpeakerStack, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, skip_chan, #TODO ask if also skip connections are used (probably not)
                                            kernel_size, padding=padding,
                                            dilation=2 ** x, norm_type=norm_type))
        mask_conv = nn.Conv1d(skip_chan, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()


    def forward(self, mixture_w):
        """
            Args:
                mixture_w (:class:`torch.Tensor`): Tensor of shape
                    [batch, n_filters, n_frames]

            Returns:
                :class:`torch.Tensor`:
                    estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = 0.
        for i in range(len(self.TCN)):
            residual, skip = self.TCN[i](output)
            output = output + residual
            skip_connection = skip_connection + skip
        score = self.mask_net(skip_connection)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score) # not needed maybe #TODO ask ?
        est_mask = est_mask / (torch.sqrt(torch.sum(est_mask**2, 2, keepdim=True))) # euclidean normalization #TODO ask
        return est_mask

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'skip_chan': self.skip_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
            'mask_act': self.mask_act
        }
        return config


class SeparationStack(nn.Module):

    def __init__(self, in_chan, n_src, spk_vec_chan=32, out_chan=None, n_blocks=8, n_repeats=3,
                 bn_chan=128, hid_chan=512, skip_chan=128, kernel_size=3,
                 norm_type="gLN", mask_act="linear"):

        super(SeparationStack, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(SepConv1DBlock(bn_chan, spk_vec_chan, hid_chan, skip_chan,
                                            kernel_size, padding=padding,
                                            dilation=2 ** x, norm_type=norm_type))
        mask_conv = nn.Conv1d(skip_chan, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w, spk_vectors):
        """
            Args:
                mixture_w (:class:`torch.Tensor`): Tensor of shape
                    [batch, n_filters, n_frames]

            Returns:
                :class:`torch.Tensor`:
                    estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = 0.
        for i in range(len(self.TCN)):
            residual, skip = self.TCN[i](output, spk_vectors) # spk vectors are used in each layer with FiLM
            output = output + residual
            skip_connection = skip_connection + skip
        score = self.mask_net(skip_connection)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)

        return est_mask

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'skip_chan': self.skip_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
            'mask_act': self.mask_act
        }
        return config



