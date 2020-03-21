from torch import nn
import torch
from asteroid.masknn import norms
from asteroid.masknn import activations
from asteroid.utils import has_arg
from asteroid.masknn.blocks import Conv1DBlock

class DilatedResidual(nn.Module):

    def __init__(self, in_chan, out_chan, kernel_size,padding, dilation, groups=1, norm_type="gLN"):
        super(DilatedResidual, self).__init__()

        self.conv = nn.Conv1d(in_chan, out_chan, kernel_size, 1, padding, dilation, groups=groups)
        self.nl = nn.PReLU()
        self.norm = norms.get(norm_type)(out_chan)

    def forward(self, x):

        out = x
        x = self.norm(self.nl(self.conv(x)))

        return x + out


class ConditionedDilatedResidual(nn.Module):

    def __init__(self, in_chan, out_chan, cond_in_chan, kernel_size, padding, dilation, groups=1, norm_type="gLN", use_FiLM=True):
        super(ConditionedDilatedResidual, self).__init__()

        self.use_FiLM = use_FiLM

        self.conv = nn.Conv1d(in_chan, out_chan, kernel_size, 1, padding, dilation, groups=groups)
        self.nl = nn.PReLU()
        self.norm = norms.get(norm_type)(out_chan)

        self.bias = nn.Linear(cond_in_chan, out_chan)
        if self.use_FiLM:
            self.mul =  nn.Linear(cond_in_chan, out_chan)

    def forward(self, x, y):
        out = x
        x = self.conv(x)
        # apply conditioning
        bias = self.bias(y.transpose(1, -1)).transpose(1, -1)
        if self.use_FiLM:
            mul = self.mul(y.transpose(1, -1)).transpose(1, -1)
            x = mul*x + bias
        else:
            x = x + bias
        x = self.norm(self.nl(x))
        return x + out


class SpeakerStack(nn.Module):

    # basically this is plain conv-tasnet remove this in future releases

    def __init__(self, in_chan, n_src, out_chan, n_blocks=8, n_repeats=1, kernel_size=3,
                 norm_type="gLN"):
        
        super(SpeakerStack, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.out_chan = out_chan

        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(DilatedResidual(in_chan, in_chan,  #TODO ask if also skip connections are used (probably not)
                                            kernel_size, padding=padding,
                                            dilation=2 ** x, norm_type=norm_type))
        mask_conv = nn.Conv1d(in_chan, n_src * out_chan , 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

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
        output = mixture_w
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output)
            output = output + residual
        output = self.mask_net(output)
        output = output.view(batch, self.n_src, self.out_chan, n_frames)
        output = output / (torch.sqrt(torch.sum(output**2, 2, keepdim=True))) # euclidean normalization #TODO ask
        return output

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
        }
        return config


class SeparationStack(nn.Module):

    # basically this is plain conv-tasnet remove this in future releases

    def __init__(self, in_chan, spk_vect_chan, out_chan=None, n_blocks=10, n_repeats=4, kernel_size=3,
                 norm_type="gLN",  mask_act="linear"):

        super(SeparationStack, self).__init__()
        self.in_chan = in_chan
        if not out_chan:
            out_chan = in_chan
        self.out_chan = out_chan
        self.spk_vect_chan = spk_vect_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act

        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(
                    ConditionedDilatedResidual(in_chan, in_chan, spk_vect_chan,  # TODO ask if also skip connections are used (probably not)
                                    kernel_size, padding=padding,
                                    dilation=2 ** x, norm_type=norm_type))

        if self.out_chan != self.in_chan:
            mask_conv = nn.Conv1d(in_chan, out_chan, 1)
            self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # output activation ?
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w, speaker_vector):
        """
            Args:
                mixture_w (:class:`torch.Tensor`): Tensor of shape
                    [batch, n_filters, n_frames]

            Returns:
                :class:`torch.Tensor`:
                    estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        output = mixture_w
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output, speaker_vector)
            output = output + residual
        if self.out_chan != self.in_chan:
            output = self.mask_net(output)
        output = self.output_act(output)
        return output

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'spk_vect_chan': self.spk_vect_chan,
            'out_chan': self.out_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'norm_type': self.norm_type,
            'mask_act': self.mask_act
        }
        return config


if __name__ == "__main__":

    a = torch.rand((3, 512, 200))
    spk_stack = SpeakerStack(512, 2, 512)
    vects = spk_stack(a)
    sep_stack = SeparationStack(512, 512)
    B, filters, frames = a.size()
    a = a[:, None, ...].expand(-1, 2, -1, -1).reshape(B*2, filters, frames)
    sep_stack(a, vects.reshape(B*2, 512, frames))
