import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
import glob

DATASET = 'WHAM'
# WHAM tasks
enh_single = {'mixture': 'mix_single',
              'sources': ['s1'],
              'infos': ['noise'],
              'default_nsrc': 1}
enh_both = {'mixture': 'mix_both',
            'sources': ['mix_clean'],
            'infos': ['noise'],
            'default_nsrc': 1}
sep_clean = {'mixture': 'mix_clean',
             'sources': ['s1', 's2'],
             'infos': [],
             'default_nsrc': 2}
sep_noisy = {'mixture': 'mix_both',
             'sources': ['s1', 's2'],
             'infos': ['noise'],
             'default_nsrc': 2}

WHAM_TASKS = {'enhance_single': enh_single,
              'enhance_both': enh_both,
              'sep_clean': sep_clean,
              'sep_noisy': sep_noisy}
# Aliases.
WHAM_TASKS['enh_single'] = WHAM_TASKS['enhance_single']
WHAM_TASKS['enh_both'] = WHAM_TASKS['enhance_both']


class WhamDataset(data.Dataset):
    """ Dataset class for WHAM source separation and speech enhancement tasks.

    Args:
        json_dir (str): The path to the directory containing the json files.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.
    """
    def __init__(self, json_dir, task, sample_rate=8000, segment=4.0,
                 nondefault_nsrc=None):
        super(WhamDataset, self).__init__()
        if task not in WHAM_TASKS.keys():
            raise ValueError('Unexpected task {}, expected one of '
                             '{}'.format(task, WHAM_TASKS.keys()))
        # Task setting
        self.json_dir = json_dir
        self.task = task
        self.task_dict = WHAM_TASKS[task]
        self.sample_rate = sample_rate
        self.seg_len = None if segment is None else int(segment * sample_rate)
        if not nondefault_nsrc:
            self.n_src = self.task_dict['default_nsrc']
        else:
            assert nondefault_nsrc >= self.task_dict['default_nsrc']
            self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        # Load json files
        mix_json = os.path.join(json_dir, self.task_dict['mixture'] + '.json')
        sources_json = [os.path.join(json_dir, source + '.json') for
                        source in self.task_dict['sources']]
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, 'r') as f:
                sources_infos.append(json.load(f))
        # Filter out short utterances only when segment is specified
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                if mix_infos[i][1] < self.seg_len:
                    drop_utt += 1
                    drop_len += mix_infos[i][1]
                    del mix_infos[i]
                    for src_inf in sources_infos:
                        del src_inf[i]

        print("Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
            drop_utt, drop_len/sample_rate/36000, orig_len, self.seg_len))
        self.mix = mix_infos
        # Handle the case n_src > default_nsrc
        while len(sources_infos) < self.n_src:
            sources_infos.append([None for _ in range(len(self.mix))])
        self.sources = sources_infos

    def __add__(self, wham):
        if self.n_src != wham.n_src:
            raise ValueError('Only datasets having the same number of sources'
                             'can be added together. Received '
                             '{} and {}'.format(self.n_src, wham.n_src))
        if self.seg_len != wham.seg_len:
            self.seg_len = min(self.seg_len, wham.seg_len)
            print('Segment length mismatched between the two Dataset'
                  'passed one the smallest to the sum.')
        self.mix = self.mix + wham.mix
        self.sources = [a + b for a, b in zip(self.sources, wham.sources)]

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start
        if self.mix[idx][1] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        # Load mixture
        x, _ = sf.read(self.mix[idx][0], start=rand_start,
                       stop=stop, dtype='float32')
        seg_len = torch.as_tensor([len(x)])
        # Load sources
        source_arrays = []
        for src in self.sources:
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((seg_len, ))
            else:
                s, _ = sf.read(src[idx][0], start=rand_start,
                               stop=stop, dtype='float32')
            source_arrays.append(s)
        sources = torch.from_numpy(np.vstack(source_arrays))
        return torch.from_numpy(x), sources


class AugmentedWhamDataset(data.Dataset):
    """ Dataset class for WHAM source separation and speech enhancement tasks.

    Args:
        wsj_train_dir (str): The path to the directory containing the wsj train/dev/test .wav files.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.
        noise_dir (str, optional): The path to the directory containing the WHAM train/dev/test .wav files
        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.
        random_gain: (tuple, optional): Minimum and maximum bounds for each source (and noise) (dB).
    """
    def __init__(self, wsj_train_dir, task, noise_dir=None, sample_rate=8000, segment=4.0,
                 nondefault_nsrc=None, random_gain=(-23, 0)):
        super(AugmentedWhamDataset, self).__init__()
        if task not in WHAM_TASKS.keys():
            raise ValueError('Unexpected task {}, expected one of '
                             '{}'.format(task, WHAM_TASKS.keys()))
        # Task setting
        self.task = task
        if self.task in ["sep_noisy", "enh_single"] and not noise_dir:
            raise RuntimeError("noise directory must be specified if task is sep_noisy or enh_single")
        self.task_dict = WHAM_TASKS[task]
        self.sample_rate = sample_rate
        self.seg_len = None if segment is None else int(segment * sample_rate)
        self.random_gain = random_gain
        if not nondefault_nsrc:
            self.n_src = self.task_dict['default_nsrc']
        else:
            assert nondefault_nsrc >= self.task_dict['default_nsrc']
            self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        # Load json files
        utterances = glob.glob(os.path.join(wsj_train_dir, "/**/*.wav"), recursive=True)
        noises = None
        if self.task in ["sep_clean", "enh_clean"]:
            noises = glob.glob(os.path.join(wsj_train_dir, "**/*.wav"), recursive=True)

        # parse utterances according to speaker
        drop_utt, drop_len = 0, 0
        print("Parsing WSJ speakers")
        examples_hashtab = {}
        for utt in utterances:
            # exclude if too short
            c_len = len(sf.SoundFile(utt))
            if  c_len < self.seg_len:
                drop_utt += 1
                drop_len += c_len
                continue
            speaker = utt.split("/")[-2]
            if speaker not in examples_hashtab.keys():
                examples_hashtab[speaker] = [(utt, c_len)]
            else:
                examples_hashtab[speaker].append((utt, c_len))

        print("Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
            drop_utt, drop_len / sample_rate / 36000, len(utterances), self.seg_len))

        examples_hashtab = {"noise": []} # not  bug will use this in __getitem__

        drop_utt, drop_len = 0, 0
        if noises:
            for noise in noises:
                c_len = len(sf.SoundFile(noise))
                if c_len < self.seg_len:
                    drop_utt += 1
                    drop_len += c_len
                    continue
                examples_hashtab["noise"].append((noise, c_len))

        print("Drop {} noises({:.2f} h) from {} (shorter than {} samples)".format(
            drop_utt, drop_len / sample_rate / 36000, len(noises), self.seg_len))

        self.examples = examples_hashtab

    def __add__(self, wham):

        raise NotImplementedError # TODO It will require different handling of other datasets, i suggets using dicts

    def __len__(self):

        return min([len(self.examples[x]) for x in self.examples.keys()])


    @staticmethod
    def get_random_subsegment(array, desired_len, tot_length):

        offset = 0
        if desired_len < tot_length:
            offset = np.random.randint(0, tot_length - desired_len)

        return array[offset: offset + desired_len]


    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Sample k speakers randomly
        c_speakers = np.random.choice([x for x in self.examples.keys() if x != "noise"], self.n_src)

        sources = []
        for i, spk in enumerate(c_speakers):
            tmp, tmp_spk_len = np.random.choice(self.examples[c_speakers[spk]], 1)
            tmp = self.get_random_subsegment(tmp, self.seg_len, tmp_spk_len)
            tmp = tmp / (np.max(np.abs(tmp)) + 1e-7)
            tmp = tmp*(10 ** (np.random.randint(self.random_gain[0], self.random_gain[1]) / 20 ) )
            sources.append(tmp)

        if self.examples["noise"]:
            # add also noise
            tmp, tmp_spk_len = np.random.choice(self.examples["noise"], 1)
            tmp = self.get_random_subsegment(tmp, self.seg_len, tmp_spk_len)
            tmp = tmp / (np.max(np.abs(tmp)) + 1e-7)
            tmp = tmp * (10 ** (np.random.randint(self.random_gain[0], self.random_gain[1]) / 20))
            sources.append(tmp)

        mix = np.mean(np.stack(sources), 0)

        if self.examples["noise"]:
            sources = sources[:-1]

        sources = np.stack(sources)

        return torch.from_numpy(mix), sources


if __name__ == "__main__":
    a = AugmentedWhamDataset("", "sep_clean", "/media/sam/Data/WSJ/wham_noise/tr/")