from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from itertools import permutations
from asteroid.losses.sdr import MultiSrcNegSDR
import math


class ClippedSDR(nn.Module):

    def __init__(self, clip_value=-30):
        super(ClippedSDR, self).__init__()

        self.snr = MultiSrcNegSDR("snr")
        self.clip_value = float(clip_value)

    def forward(self, est_targets, targets):

        return torch.clamp(self.snr(est_targets, targets), min=self.clip_value)


class SpeakerVectorLoss(nn.Module):

    def __init__(self, n_speakers, embed_dim=32, learnable_emb=True, loss_type="global",
                 weight=10, distance_reg=0.3, gaussian_reg=0.2):
        super(SpeakerVectorLoss, self).__init__()

        self.loss_type = loss_type
        self.weight = float(weight)
        self.distance_reg = float(distance_reg)
        self.gaussian_reg = float(gaussian_reg)

        assert loss_type in ["distance", "global", "local"]

        spk_emb = torch.rand((n_speakers, embed_dim))  # generaate points in n-dimensional unit sphere
        norms = torch.sum(spk_emb ** 2, -1, keepdim=True).sqrt()
        # c = torch.rand((n_speakers + 1, embed_dim))**(1/3)
        spk_emb = spk_emb / norms

        if learnable_emb == True:
            self.spk_embeddings = nn.Parameter(spk_emb)
        else:
            self.register_buffer("spk_embeddings", spk_emb)

        if loss_type != "dist":
            self.alpha = nn.Parameter(torch.Tensor([1.]))
            self.beta = nn.Parameter(torch.Tensor([0.]))

    @staticmethod
    def _l_dist_speaker(c_spk_vec_perm, spk_embeddings, spk_labels, spk_mask):

        utt_embeddings = spk_embeddings[spk_labels].unsqueeze(-1) * spk_mask.unsqueeze(2)
        c_spk = c_spk_vec_perm[:, 0]
        pair_dist = ((c_spk.unsqueeze(1) - c_spk_vec_perm)**2).mean(2)
        pair_dist = pair_dist[:, 1:]
        distance = ((c_spk_vec_perm - utt_embeddings)**2).mean(2)
        return (distance + F.relu(1. - pair_dist).mean(1).unsqueeze(1)).mean(1)

    def _l_local_speaker(self, c_spk_vec_perm, spk_embeddings, spk_labels, spk_mask):

        utt_embeddings = spk_embeddings[spk_labels].unsqueeze(-1) * spk_mask.unsqueeze(2)
        alpha = torch.clamp(self.alpha, 1e8)

        distance = alpha*((c_spk_vec_perm - utt_embeddings)**2).mean(2) + self.beta
        out = -distance - torch.log(torch.exp(-distance).mean(1)).unsqueeze(1)

        return out.mean(1)

    def _l_global_speaker(self, c_spk_vec_perm, spk_embeddings, spk_labels, spk_mask):

        utt_embeddings = spk_embeddings[spk_labels].unsqueeze(-1) * spk_mask.unsqueeze(2)
        alpha = torch.clamp(self.alpha, 1e-8)

        distance_utt = alpha*((c_spk_vec_perm - utt_embeddings)**2).mean(2) + self.beta

        B, src, embed_dim, frames = c_spk_vec_perm.size()
        spk_embeddings = spk_embeddings.reshape(1, spk_embeddings.shape[0], embed_dim, 1).expand(B, -1, -1, frames)
        distances = alpha * ((c_spk_vec_perm.unsqueeze(1) - spk_embeddings.unsqueeze(2)) ** 2).mean(3) + self.beta
        out = -distance_utt - torch.log(torch.exp(-distances).mean(1))
        return out.mean(1)


    def forward(self, speaker_vectors, spk_mask, spk_labels):

        if self.gaussian_reg:
            noise = torch.randn(self.spk_embeddings.size(), device=speaker_vectors.device)*math.sqrt(self.gaussian_reg)
            spk_embeddings = self.spk_embeddings + noise
        else:
            spk_embeddings = self.spk_embeddings

        if self.distance_reg:

            pairwise_dist = ((self.spk_embeddings.unsqueeze(0) - self.spk_embeddings.unsqueeze(1))**2).mean(-1)
            idx = torch.arange(0, pairwise_dist.shape[0])
            pairwise_dist[idx, idx] = np.inf # masking
            distance_reg = -torch.min(torch.log(pairwise_dist), dim=0)[0]

        # speaker vectors B, n_src, dim, frames
        # spk mask B, n_src, frames boolean mask
        # spk indxs list of len B of list which contains spk label for current utterance
        B, n_src, embed_dim, frames = speaker_vectors.size()

        n_src = speaker_vectors.shape[1]
        perms = list(permutations(range(n_src)))
        if self.loss_type == "distance":
            loss_set = torch.stack([self._l_dist_speaker(speaker_vectors[:, perm], spk_embeddings, spk_labels, spk_mask) for perm in perms],
                                   dim=1)
        elif self.loss_type == "local":
            loss_set = torch.stack([self._l_local_speaker(speaker_vectors[:, perm], spk_embeddings, spk_labels, spk_mask) for perm in perms],
                                   dim=1)
        else:
            loss_set = torch.stack([self._l_global_speaker(speaker_vectors[:, perm], spk_embeddings, spk_labels, spk_mask) for perm in perms],
                                   dim=1)

        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)

        # reorder sources for each frame !!
        perms = min_loss.new_tensor(perms, dtype=torch.long)
        perms = perms[..., None, None].expand(-1, -1, B, frames)
        min_loss_idx = min_loss_idx[None, None,...].expand(1, n_src, -1, -1)
        min_loss_perm = torch.gather(perms, dim=0, index=min_loss_idx)[0]
        min_loss_perm = min_loss_perm.transpose(0, 1).reshape(B, n_src, 1, frames).expand(-1, -1, embed_dim, -1)
        # tot_loss

        spk_loss = self.weight*min_loss.mean()
        if self.distance_reg:
            spk_loss += self.distance_reg*distance_reg.mean()
        reordered_sources = torch.gather(speaker_vectors, dim=1, index=min_loss_perm)
        return spk_loss, reordered_sources


if __name__ == "__main__":

    loss_spk = SpeakerVectorLoss(1000, 32, loss_type="distance") # 1000 speakers in training set

    speaker_vectors = torch.rand(2, 3, 32, 200)
    speaker_labels = torch.from_numpy(np.array([[1, 2, 0], [5, 2, 10]]))
    speaker_mask = torch.randint(0, 2, (2, 3, 200)) # silence where there are no speakers actually thi is test
    speaker_mask[:, -1, :] = speaker_mask[:, -1, :]*0
    loss_spk(speaker_vectors, speaker_mask, speaker_labels)


    c = ClippedSDR(-30)
    a = torch.rand((2, 3, 200))
    print(c(a, a))














