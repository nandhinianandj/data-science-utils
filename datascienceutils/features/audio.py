import numpy as np
import os
import pandas as pd
import python_speech_features as psf

def mfcc_features(input_sound):
    import scipy.io.wavfile as wav
    (rate, signal) = wav.read(input_sound)
    mfcc_feat = pd.DataFrame(psf.mfcc(signal, rate))
    return mfcc_feat

def filter_bank_energies(input_sound):
    import scipy.io.wavfile as wav
    (rate, signal) = wav.read(input_sound)
    fbank_feat = pd.DataFrame(psf.fbank(signal, rate))
    return fbank_feat

def spectral_subband_centroids(input_sound):
    import scipy.io.wavfile as wav
    (rate, signal) = wav.read(input_sound)
    ssc_feat = pd.DataFrame(psf.ssc(signal, rate))
    return ssc_feat

def all_features(input_sound):
    res = dict()
    res['mfcc'] = mfcc_features(input_sound)
    res['ssc'] = spectral_subband_centroids(input_sound)
    res['fbank'] = filter_bank_energies(input_sound)
    return res

