import torch
from torch.utils import data
from asteroid.engine.losses import PITLossContainer
from asteroid.filterbanks.inputs_and_masks import take_mag


class Dataset(data.Dataset):
    def __init__(self, encoder, n_freq=257):
        self.encoder = encoder
        self.n_freq = n_freq

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        s1 = torch.randn(1, 32000)
        s2 = torch.randn(1, 32000)
        mixture = s1 + s2
        return mixture, torch.stack([s1, s2]), {'mixture': mixture}


def loss_func(targets, est_masks, mixture=None, encoder=None):
    """

    Args:
        targets: time domain signals (batch, nsrc, time)
        est_masks: TF estimates (batch, nsrc, freq, time)
        mixture: time domain signal (batch, time)
        encoder:
    Returns:
        The pairwise losses on masks (batch, nsrc, nsrc)
    """
    # import ipdb; ipdb.set_trace()
    eps = 1e-8
    # Forward each time domain target through the encoder + take magnitude.
    # (batch, nsrc, freq, time)
    source_tfs = torch.stack([take_mag(encoder(s.unsqueeze(1))) for s in
                              targets.transpose(0, 1)], dim=1)
    # (batch, 1, freq, time)
    denominator = torch.sum(source_tfs, dim=1, keepdim=True) + eps
    # (batch, nsrc, freq, time)
    masks = source_tfs / denominator
    # (batch, 1, nsrc, freq, time) and (batch, nsrc, 1, freq, time)
    masks = masks.unsqueeze(1)
    est_masks = est_masks.unsqueeze(2)
    # Compute MSE between mask and estimate
    # (batch, nscr, nsrc, freq, time)
    pw_batch_loss = torch.mean((masks - est_masks).pow(2), dim=(-1, -2))
    # Compute MSE between mask and estimated mask weighted by the mixture
    mixt_tf = take_mag(encoder(mixture))[:, None, None]
    pw_batch_loss_2 = torch.mean((mixt_tf * (masks - est_masks)).pow(2),
                            dim=(-1, -2))
    # Compute MSE between estimated mask applied to mixture and targets.
    pw_batch_loss_3 = torch.mean(
        (mixt_tf * est_masks - source_tfs.unsqueeze(1)).pow(2), dim=(-1, -2)
    )
    return pw_batch_loss



if __name__ == '__main__':
    from asteroid.filterbanks import Encoder, STFTFB
    n_src = 2
    encoder = Encoder(STFTFB(n_filters=256, kernel_size=256, stride=128))
    loss_cont = PITLossContainer(loss_func, n_src=n_src)


    batch_size = 10
    time_len = 32000
    mixture = torch.randn(batch_size, 1, time_len)
    mix_enc = encoder(mixture)
    freq_dim = take_mag(mix_enc).shape[1]
    freq_time = mix_enc.shape[-1]

    targets = torch.randn(batch_size, n_src, time_len)
    est_masks = torch.randn(batch_size, n_src, freq_dim, freq_time)

    loss_1 = loss_func(targets, est_masks, mixture=mixture, encoder=encoder)
    infos_dict = {'mixture': mixture, 'encoder': encoder}
    loss_2 = loss_cont.compute(targets, est_masks, infos=infos_dict)
    # assert loss_1 == loss_2

