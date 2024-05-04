

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Loads data from Excel file
# data = pd.read_excel("engine2_normalized.xlsx")

# # Selects relevant signals (Engine Speed and Engine Load)
# engine_speed = data['engine_speed']
# engine_load = data['engine_load']

# # Plots signals
# plt.figure(figsize=(10, 6))
# plt.plot(engine_speed, label='Engine Speed')
# plt.plot(engine_load, label='Engine Load')
# plt.title('Engine Signals')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.show()

# #  analysis (calculates mean, min, max)
# def calculate_mean(signal):
#     return sum(signal) / len(signal)

# def moving_average(signal, window_size):
#     smoothed_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
#     return smoothed_signal

# # Define a function for median filtering
# def median_filter(signal, window_size):
#     smoothed_signal = signal.copy()
#     half_window = window_size // 2
#     for i in range(half_window, len(signal) - half_window):
#         smoothed_signal[i] = np.median(signal[i - half_window:i + half_window])
#     return smoothed_signal

# # Defines a function for low-pass filtering
# def low_pass_filter(signal, cutoff_frequency, sampling_rate):
#     nyquist_frequency = 0.5 * sampling_rate
#     normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency
#     b, a = signal.butter(4, normalized_cutoff_frequency, btype='low', analog=False)
#     filtered_signal = signal.filtfilt(b, a, signal, padtype='constant')
#     return filtered_signal
# # Fourier Transform and Frequency Analysis
# def frequency_analysis(signal, sampling_rate):
#     n = len(signal)
#     fft_result = np.fft.fft(signal)
#     freq = np.fft.fftfreq(n, d=1/sampling_rate)
#     amplitude_spectrum = np.abs(fft_result)
#     return freq[:n//2], amplitude_spectrum[:n//2]

# engine_speed_mean = calculate_mean(engine_speed)
# engine_load_mean = calculate_mean(engine_load)

# print("Mean Engine Speed:", engine_speed_mean)
# print("Mean Engine Load:", engine_load_mean)


# # Apply moving average filtering
# engine_speed_smoothed_ma = moving_average(engine_speed, window_size=10)
# engine_load_smoothed_ma = moving_average(engine_load, window_size=10)

# # Apply median filtering
# engine_speed_smoothed_median = median_filter(engine_speed, window_size=10)
# engine_load_smoothed_median = median_filter(engine_load, window_size=10)

# # Plot original and smoothed signals
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(engine_speed, label='Original Engine Speed', alpha=0.5)
# plt.plot(engine_speed_smoothed_ma, label='Moving Average Filtered', linestyle='--', color='red')
# plt.plot(engine_speed_smoothed_median, label='Median Filtered', linestyle='--', color='green')
# plt.title('Engine Speed Signals with Moving Average and Median Filtering')
# plt.xlabel('Time')
# plt.ylabel('Engine Speed')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(engine_load, label='Original Engine Load', alpha=0.5)
# plt.plot(engine_load_smoothed_ma, label='Moving Average Filtered', linestyle='--', color='red')
# plt.plot(engine_load_smoothed_median, label='Median Filtered', linestyle='--', color='green')
# plt.title('Engine Load Signals with Moving Average and Median Filtering')
# plt.xlabel('Time')
# plt.ylabel('Engine Load')
# plt.legend()

# plt.tight_layout()
# plt.show()


# # Sampling rate (assuming uniform sampling)
# sampling_rate = 1  # Hz

# # Frequency analysis for engine speed
# engine_speed_freq, engine_speed_spectrum = frequency_analysis(engine_speed, sampling_rate)

# # Frequency analysis for engine load
# engine_load_freq, engine_load_spectrum = frequency_analysis(engine_load, sampling_rate)

# # Plot frequency spectra
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(engine_speed_freq, engine_speed_spectrum)
# plt.title('Engine Speed Frequency Spectrum')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(engine_load_freq, engine_load_spectrum)
# plt.title('Engine Load Frequency Spectrum')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.grid(True)

# plt.tight_layout()
# plt.show()
