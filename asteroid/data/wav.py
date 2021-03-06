"""
Utils used to handle a wav file
@author: Sunit Sivasankaran, Inria-Nancy
"""

import os
import numpy as np
import soundfile as sf
import ipdb


class SingleWav(object):
    """
    Interface to handle a single wav file

    # Arguments
        file_name: The path to the wav file
        channel_interest: An array of interested channels.
            Used in case of multichannel signals
        wav_id: An id to identify the wav file
        save: Boolean, Save the data untill the object is destroyed if True

    # Example
        SingleWav('/home/test.wav')

    """
    def __init__(self, file_name, channel_interest=None, \
            wav_id=None, save=False):
        self.file_name = file_name
        self.__wav_data = None
        self.__id = wav_id
        self.info = None
        self.sampling_rate = None
        self.sample_len = None
        self.channel_count = None
        self.save = save
        self.channel_interest = None
        self.verify()
        if channel_interest is not None:
            self.channel_interest = np.array(channel_interest)

    def verify(self):
        """
        Verify if all the information is good
        """
        assert os.path.exists(self.file_name),\
            self.file_name +' does not exists'


    def update_info(self):
        """
            Get wav related info and place it in the
            `info` variable.
            Note: Avoid calling this in the `__init__` section. Very time
                consuming
        """
        if self.info is None:
            self.info = sf.info(self.file_name)
            self.sampling_rate = self.info.samplerate
            self.sample_len = int(self.info.samplerate * self.info.duration)
            self.channel_count = self.info.channels

    @property
    def wav_len(self):
        """
            Get the sample length of the signal
        #Returns
            A number of samples in wav
        """
        if self.sample_len is None:
            self.update_info()
        return self.sample_len

    @property
    def data(self):
        """
            Read the wav file if not saved
        #Returns
            A two dimensional numpy array of shape [samples, channels]
        """
        self.update_info()
        if self.__wav_data is not None:
            return self.__wav_data
        #print('reading ', self.file_name)
        wav_data, self.sampling_rate = \
                    sf.read(self.file_name, always_2d=True)
        if self.channel_interest is not None:
            wav_data = wav_data[:, self.channel_interest]
        if self.save:
            self.__wav_data = wav_data
        return wav_data

    # Handle with statement
    def __enter__(self):
        self.__wav_data = self.data

    def __exit__(self, data_type, data_val, data_tb):
        if not self.save:
            self.__wav_data = None

    def save_space(self):
        """ Remove the saved data. self.data will read from the file again.
        """
        self.__wav_data = None

    def temp_save(self):
        """ Temporarily save the wav data. Call `save_space` to remove it.
        """
        self.__wav_data = self.data


    @property
    def wav_id(self):
        """getter for the wav id
        """
        return self.__id

    @wav_id.setter
    def wav_id(self, value):
        self.__id = value

    def write_wav(self, path):
        """ Write the wav data into an other path
        """
        sf.write(path, self.data, self.sampling_rate)

class MultipleWav(SingleWav):
    """
        Handle a set of wav files as a single object.
    # Arguments
        file_name_list: A list of wav file names
        channel_interest: An array of interested channels.
            Used in case of multichannel signals
        wav_id: An id to identify the bunch of wav file
        save: Boolean, Save the data untill the object is destroyed if True

    """
    def __init__(self, file_name_list, channel_interest=None, \
            wav_id=None, save=False):
        self.file_name_list = file_name_list
        self.__wav_data = None
        self.__id = wav_id
        self.info = None
        self.sampling_rate = None
        self.sample_len = None
        self.channel_count = None
        self.info_list = []
        self.sampling_rate_list = []
        self.sample_len_list = []
        self.channel_count_list = []
        self.save = save
        self.channel_interest = None
        if channel_interest is not None:
            self.channel_interest = np.array(channel_interest)

    def update_info(self):
        if self.info is None:
            for _file_ in self.file_name_list:
                info = sf.info(_file_)
                self.info_list.append(info)
                self.sampling_rate_list.append(info.samplerate)
                self.sample_len_list.append(int(info.samplerate * info.duration))
                self.channel_count_list.append(info.channels)
            self.info = info
            self.sampling_rate = info.samplerate
            self.sample_len = int(info.samplerate * info.duration)
            self.channel_count = info.channels

    @property
    def data(self):
        """Reads all the files in the file list
        #Returns
            A list of wav signals
        """
        self.update_info()
        if self.__wav_data is not None:
            return self.__wav_data
        #print('reading ', self.file_name)
        wav_data = []
        for _file_ in self.file_name_list:
            _wav_data, _ = sf.read(_file_, always_2d=True)
            if self.channel_interest is not None:
                _wav_data = _wav_data[:, self.channel_interest]
            wav_data.append(_wav_data)
        if self.save:
            self.__wav_data = wav_data
        return wav_data
