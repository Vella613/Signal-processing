import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pydub import AudioSegment
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
import settings

class Functions:
       # Sampling rate (assuming uniform sampling)
    sampling_rate = 1  # Hz
    @staticmethod
    def preprocess_data(signal):
        # Applys moving average filtering
        smoothed_signal = Functions.moving_average(signal, window_size=10)
        # Applys median filtering
        smoothed_signal = Functions.median_filter(smoothed_signal, window_size=10)
        # Applys low-pass filtering
        smoothed_signal = Functions.low_pass_filter(smoothed_signal, cutoff_frequency=0.1, sampling_rate=Functions.sampling_rate)
        return smoothed_signal

    
    @staticmethod
    def frequency_analysis(signal, sampling_rate, precision=64):
        n = len(signal)
        
        # Sets precision for calculations
        np.set_printoptions(precision=precision)

        # Converts signal to higher precision
        signal = np.array(signal, dtype=np.float64)

        fft_result = np.fft.fft(signal)
        freq = np.fft.fftfreq(n, d=1/sampling_rate)

        amplitude_spectrum = np.abs(fft_result)

        print("FFT Result:", fft_result)
        print("Frequencies:", freq)
        print("Amplitude Spectrum:", amplitude_spectrum)

        return freq[:n//2], amplitude_spectrum[:n//2]

    @staticmethod
    def moving_average(signal, window_size):
        smoothed_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        return smoothed_signal

    @staticmethod
    def median_filter(signal, window_size):
        smoothed_signal = signal.copy()
        half_window = window_size // 2
        for i in range(half_window, len(signal) - half_window):
            smoothed_signal[i] = np.median(signal[i - half_window:i + half_window])
        return smoothed_signal

    @staticmethod
    def low_pass_filter(signal, cutoff_frequency, sampling_rate):
        nyquist_frequency = 0.5 * sampling_rate
        normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency
        b, a = butter(4, normalized_cutoff_frequency, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal, padtype='constant')
        return filtered_signal

 
