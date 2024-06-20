import numpy as np
import pandas as pd
import librosa
import pywt
from scipy.signal import spectrogram
from matplotlib.pyplot import specgram
from sklearn.impute import SimpleImputer
from feature_engine.imputation import MeanMedianImputer

import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from class_functions import Functions
from Regression_class import LinearRegression
from scipy import signal
from scipy.fft import fftshift
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

def  main():
    data = pd.read_excel("drehzahl_daten_3.xlsx")
    data = data.interpolate(method='linear')


    # data = data.dropna(subset=['Drezahl']) 


    engine_speed = data['engine_speed']
    time = data['time']

    plt.figure(figsize=(10, 6))
    plt.plot(engine_speed, label='Engine Speed')
    plt.title('Engine Signals')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    'Normalisierung der Daten nicht notwendig!!!'

    # engine_speed_normalized = (engine_speed - engine_speed.mean()) / engine_speed.std()

    
    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(engine_speed, label='Original Engine Speed', alpha=0.5)
    # plt.plot(engine_speed_normalized, label='Normalized', linestyle='--', color='red')
    # plt.title('Engine Speed Signals normalized')
    # plt.xlabel('Time')
    # plt.ylabel('Engine Speed')
    # plt.legend()

    # engine_load_normalized = (engine_load - engine_load.mean()) / engine_load.std()

    # Applying logarithmic transformation to emphasize small variations
    # engine_speed_log = np.log(engine_speed_normalized.abs() + 1)
    # engine_load_log = np.log(engine_load_normalized.abs() + 1)


    # moving average filtering
    engine_speed_smoothed_ma = Functions.moving_average(engine_speed, window_size=10)
    # engine_load_smoothed_ma = Functions.moving_average(engine_load, window_size=10)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(engine_speed, label='Original Engine Speed', alpha=0.5)
    plt.plot(engine_speed_smoothed_ma, label='Moving Average Filtered', linestyle='--', color='red')
    plt.title('Engine Speed Signals with Moving Average')
    plt.xlabel('Time')
    plt.ylabel('Engine Speed')
    plt.legend()






    #  median filtering
    engine_speed_smoothed_median = Functions.median_filter(engine_speed, window_size=10)
    # engine_load_smoothed_median = Functions.median_filter(engine_load, window_size=10)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(engine_speed, label='Original Engine Speed', alpha=0.5)
    plt.plot(engine_speed_smoothed_median, label='Median Filtered', linestyle='--', color='green')
    plt.title('Engine Speed Signals with Median Filtering')
    plt.xlabel('Time')
    plt.ylabel('Engine Speed')
    plt.legend()


    # Plots original and smoothed signals
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(engine_speed, label='Original Engine Speed', alpha=0.5)
    plt.plot(engine_speed_smoothed_ma, label='Moving Average Filtered', linestyle='--', color='red')
    plt.plot(engine_speed_smoothed_median, label='Median Filtered', linestyle='--', color='green')
    plt.title('Engine Speed Signals with Moving Average and Median Filtering')
    plt.xlabel('Time')
    plt.ylabel('Engine Speed')
    plt.legend()


    plt.tight_layout()
    plt.show()

    
    engine_speed_preprocessed_signal  = Functions.preprocess_data(engine_speed)

    
    plt.subplot(2, 1, 1)
    plt.plot(engine_speed_preprocessed_signal, label='Signal after moving average, median filter and low pass filter')
    plt.title('After low pass filter')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # preprocess_data function and low-pass filter might be unnecessary

    # Calculating mean engine speed and engine load after preprocessing
    engine_speed_mean = engine_speed.mean()
    # engine_load_mean = engine_load.mean()

    print("Mean Engine Speed:", engine_speed_mean)
    # print("Mean Engine Load:", engine_load_mean)
  

    '---------------------------Spectrum----------------------------------------------------------'

    # Frequency analysis for engine speed
    engine_speed_freq, engine_speed_spectrum = Functions.frequency_analysis(engine_speed, Functions.sampling_rate)

    # Frequency analysis for engine load
    # engine_load_freq, engine_load_spectrum = Functions.frequency_analysis(engine_load, Functions.sampling_rate)

    # Plot frequency spectra
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)

    plt.plot(engine_speed_freq, engine_speed_spectrum)
    plt.title('Engine Speed Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)


    plt.tight_layout()
    plt.show()

    print("Engine Speed after Preprocessing:", engine_speed)
    # print("Engine Load after Preprocessing:", engine_load)

    print("Engine_speed after preprocessing:", engine_speed)
    # print("Engine_load after preprocessing:", engine_load)


    print("Original engine speed:", engine_speed[:10])  


    '---------------------Train and evaluate Model --for Predictions-------------------------'

    # adjusts noise_level parameter as needed
    existing_engine_speed = data['engine_speed']  

    # trains and evaluates the model for engine speed
    model_engine_speed = Functions.train_and_evaluate_model(data, 'engine_speed')  


    # generates synthetic engine speed and engine load data
    new_engine_speed = Functions.generate_new_engine_speed(existing_engine_speed)

    # creates a DataFrame for the new synthetic data
    new_data = pd.DataFrame({
        'engine_speed': new_engine_speed,

    })

        # Calculate variance, skewness, and kurtosis for engine_speed
    variance_engine_speed = np.var(engine_speed)
    skewness_engine_speed = pd.Series(engine_speed).skew()
    kurtosis_engine_speed = pd.Series(engine_speed).kurtosis()


    print("Engine Speed Variance:", variance_engine_speed)
    print("Engine Speed Skewness:", skewness_engine_speed)
    print("Engine Speed Kurtosis:", kurtosis_engine_speed)



    '??????-------------------------------- not working properly idk why------------------------?????????????????'
    # engine_speed_np = engine_speed.values.astype(np.float32)  # Convert to numpy array

    # # Check data range and statistics
    # print("Data Mean:", np.mean(engine_speed_np))
    # print("Data Std Dev:", np.std(engine_speed_np))
    # print("Data Min:", np.min(engine_speed_np))
    # print("Data Max:", np.max(engine_speed_np))

    # # Normalize the data (optional but can help with visualization)
    # engine_speed_np = (engine_speed_np - np.mean(engine_speed_np)) / np.std(engine_speed_np)

    # # Parameters for STFT
    # n_fft = 2048  # You may adjust this based on your signal length and resolution needs
    # hop_length = n_fft // 4  # Common choice for hop_length

    # # Compute STFT
    # engine_speed_stft = librosa.stft(engine_speed_np, n_fft=n_fft, hop_length=hop_length)

    # # Plot STFT spectrogram
    # plt.figure(figsize=(10, 6))
    # librosa.display.specshow(librosa.amplitude_to_db(np.abs(engine_speed_stft), ref=np.max),
    #                         sr=Functions.sampling_rate, hop_length=hop_length, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Engine Speed STFT Spectrogram')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.tight_layout()
    # plt.show()
    # STFT for engine_speed

    '-----------------------------------Spectrogram----------------------------------'

    engine_speed_np = engine_speed.values.astype(np.float32)  # Convert to numpy array
    # Set parameters for spectrogram computation

    fs = Functions.sampling_rate  # Sampling frequency
    nperseg = 256  # Length of each segment
    noverlap = 128  # Overlap between segments

    # nfft = min(2048, len(engine_speed_np))  # Choose nfft dynamically based on signal length

    signal_length = len(engine_speed)

    # Choose an appropriate NFFT value, for example, half of the signal length
    nfft = min(2048, signal_length // 2)
  
    # f, t, Sxx = signal.spectrogram(engine_speed, fs, return_onesided=False)
    print('length of engine speed signal',len(engine_speed))

    Functions.plot_spectrogram(time,data,fs)

    # plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()

    plt.show()
        # while(1):
    #     pass

    '--------------------------------Features Extraction------------------'

    # # Extract features from the engine_speed series
    # features = extract_engine_speed_features(engine_speed)
    # print(features)
    # Create a MeanMedianImputer object
    imputer = MeanMedianImputer(imputation_method='mean')

    # Fit the imputer to the data and transform it
    imputed_data = imputer.fit_transform(data)

    # Extract features from the imputed data
    features = imputed_data.describe()

    print(features)
    '---------------------Wavelet Transform for engine_speed---------------------'

    coeffs, freqs = pywt.cwt(engine_speed, scales=np.arange(1, 128), wavelet='morl')

    # Plot Wavelet coefficients
    plt.figure(figsize=(10, 6))
    plt.imshow(coeffs, extent=[0, len(engine_speed), 1, 128], cmap='PRGn', aspect='auto',
            vmax=abs(coeffs).max(), vmin=-abs(coeffs).max())
    plt.colorbar()
    plt.title('Engine Speed Wavelet Transform Coefficients')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.tight_layout()
    plt.show()

    '------------------------Generated data--------------------------------------------'

    # Plots original and generated engine speed
    plt.figure(figsize=(10, 6))
    plt.plot(existing_engine_speed, label='Original Engine Speed',  color='blue', alpha=0.5)
    plt.plot(new_engine_speed, label='Generated Engine Speed',  color='green', alpha=0.5)
    plt.title('Original and Generated Engine Speed')
    plt.xlabel('Time')
    plt.ylabel('Engine Speed')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    'I removed the engine load part since it looks like it is not important for the engine speed detection'



if __name__ == "__main__":
    main()