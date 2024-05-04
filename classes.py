import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class Functions:
    def preprocess_data(signal):
        # Apply moving average filtering
        smoothed_signal = np.moving_average(signal, window_size=10)
        # Apply median filtering
        smoothed_signal = plt.median_filter(smoothed_signal, window_size=10)
        # Apply low-pass filtering
        smoothed_signal = plt.low_pass_filter(smoothed_signal, cutoff_frequency=0.1, sampling_rate=sampling_rate)
        return smoothed_signal

    def frequency_analysis(signal, sampling_rate, precision=64):
        n = len(signal)
        
        # Set precision for calculations
        np.set_printoptions(precision=precision)

        # Convert signal to higher precision
        signal = np.array(signal, dtype=np.float64)

        fft_result = np.fft.fft(signal)
        freq = np.fft.fftfreq(n, d=1/sampling_rate)

        amplitude_spectrum = np.abs(fft_result)

        print("FFT Result:", fft_result)
        print("Frequencies:", freq)
        print("Amplitude Spectrum:", amplitude_spectrum)

        return freq[:n//2], amplitude_spectrum[:n//2]
    # # Fourier Transform and Frequency Analysis
    # def frequency_analysis(signal, sampling_rate):
    #     n = len(signal)
    #     fft_result = np.fft.fft(signal)
    #     freq = np.fft.fftfreq(n, d=1/sampling_rate)
    #     amplitude_spectrum = np.abs(fft_result)

    #     print("FFT Result:", fft_result)
    #     print("Frequencies:", freq)
    #     print("Amplitude Spectrum:", amplitude_spectrum)

    #     return freq[:n//2], amplitude_spectrum[:n//2]

    # Sampling rate (assuming uniform sampling)
    sampling_rate = 1  # Hz

    # Apply moving average filtering
    def moving_average(signal, window_size):
        smoothed_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        return smoothed_signal

    # Apply median filtering
    def median_filter(signal, window_size):
        smoothed_signal = signal.copy()
        half_window = window_size // 2
        for i in range(half_window, len(signal) - half_window):
            smoothed_signal[i] = np.median(signal[i - half_window:i + half_window])
        return smoothed_signal

    # Define a function for low-pass filtering
    def low_pass_filter(signal, cutoff_frequency, sampling_rate):
        nyquist_frequency = 0.5 * sampling_rate
        normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency
        b, a = butter(4, normalized_cutoff_frequency, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal, padtype='constant')
        return filtered_signal
