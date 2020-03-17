from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from itertools import permutations
from asteroid.losses.sdr import MultiSrcNegSDR


class ClippedSDR(nn.Module):

    def __init__(self, clip_value=-30):
        super(ClippedSDR, self).__init__()

        self.snr = MultiSrcNegSDR("snr")
        self.clip_value = float(clip_value)

    def forward(self, est_targets, targets):

        return torch.clamp(self.snr(est_targets, targets), min=self.clip_value)


class SpeakerVectorLoss(nn.Module):

    def __init__(self, n_speakers, embed_dim=32):
        super(SpeakerVectorLoss, self).__init__()

        # paper not entirely clear my workaround use an explicit embedding for silence.
        self.spk_embeddings = nn.Parameter(torch.rand((n_speakers+1, embed_dim)))
        self.alpha = nn.Parameter(torch.Tensor([1.]))
        self.beta =  nn.Parameter(torch.Tensor([0.]))

    @staticmethod
    def _l_dist_speaker(c_spk_vec_perm, c_embeddings):

        c_spk_vec = c_spk_vec_perm[:, 0]
        pair_dist = torch.mean(c_spk_vec.unsqueeze(1)**2 - c_spk_vec_perm[:, 1:]**2, (1, 2))

        distance = torch.mean(c_spk_vec_perm**2 - c_embeddings**2, (1, 2)) # average over embed dim
        return distance + F.relu(1. - pair_dist)

    def _l_local_speaker(self, c_spk_vec_perm, c_embeddings): #TODO

        distance = self.alpha*torch.mean(c_spk_vec_perm**2 - c_embeddings**2, (1, 2)) + self.beta
        local = -F.log_softmax(-distance, dim=1).mean(2)
        raise NotImplementedError

        return -distance +local

    def _l_global_speaker(self, c_spk_vec_perm): #TODO
        raise NotImplementedError

    def _l_speaker(self, c_spk_vec_perm, spk_mask, spk_labels):

        # get current embeddings from labels
        c_embeddings = self.spk_embeddings[spk_labels].unsqueeze(-1)*spk_mask.unsqueeze(2)
        return SpeakerVectorLoss._l_dist_speaker(c_spk_vec_perm, c_embeddings)

    def forward(self, speaker_vectors, spk_mask, spk_labels):

        # speaker vectors B, n_src, dim, frames
        # spk mask B, n_src, frames boolean mask
        # spk indxs list of len B of list which contains spk label for current utterance

        # TODO memoize intra speaker vector distances maybe to be computed ony once
        B, n_src, embed_dim, frames = speaker_vectors.size()

        n_src = speaker_vectors.shape[1]
        perms = list(permutations(range(n_src)))
        loss_set = torch.stack([self._l_speaker(speaker_vectors[:, perm], spk_mask,
                                                spk_labels) for perm in perms],
                               dim=1)
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)

        # reorder sources for each frame !!
        perms = min_loss.new_tensor(perms, dtype=torch.long)
        perms = perms[..., None, None].expand(-1, -1, B, frames)
        min_loss_idx = min_loss_idx[None, None,...].expand(1, n_src, -1, -1)
        min_loss_perm = torch.gather(perms, dim=0, index=min_loss_idx)[0]
        min_loss_perm = min_loss_perm.transpose(0, 1).reshape(B, n_src, 1, frames).expand(-1, -1, embed_dim, -1)

        reordered_sources = torch.gather(speaker_vectors, dim=1, index=min_loss_perm)
        return min_loss, reordered_sources

if __name__ == "__main__":
    loss_spk = SpeakerVectorLoss(1000, 32) # 1000 speakers in training set

    speaker_vectors = torch.rand(2, 3, 32, 200)
    speaker_labels = torch.from_numpy(np.array([[1, 2, 0], [5, 2, 10]]))
    speaker_mask = torch.randint(0, 2, (2, 3, 200)) # silence where there are no speakers actually thi is test
    speaker_mask[:, -1, :] = speaker_mask[:, -1, :]*0
    loss_spk(speaker_vectors, speaker_mask, speaker_labels)


    c = ClippedSDR(-30)
    a = torch.rand((2, 3, 200))
    print(c(a, a))














