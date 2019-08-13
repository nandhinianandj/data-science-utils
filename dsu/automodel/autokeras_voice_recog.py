# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : autokeras_voice_recog.py
#
#* Purpose :
#
#* Creation Date : 31-07-2019
#
#* Last Modified : Wednesday 31 July 2019 08:32:53 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

from autokeras.constant import Constant
import torchaudio
import scipy.signal
import librosa
import torch
import numpy as np

def load_audio(path):
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[0] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=0)  # multiple channels, average
    return sound


class SpectrogramParser:
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = scipy.signal.hamming
        self.normalize = normalize
        self.augment = augment
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        y = load_audio(audio_path)

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, _ = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

parser = SpectrogramParser(Constant.VOICE_RECONGINIZER_AUDIO_CONF, normalize=True)
spect = parser.parse_audio("test.wav").contiguous()


### Predict voice
from autokeras import VoiceRecognizer
voice_recognizer = VoiceRecognizer()
print(voice_recognizer.predict(audio_data=spect))

