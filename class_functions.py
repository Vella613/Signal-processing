import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# from pydub import AudioSegment
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from Regression_class import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy import signal
from scipy.fft import fftshift
from scipy.fft import fft, fftfreq

import io

class Functions:
    sampling_rate = 10  # Hz

    
    @staticmethod
    def median_filter(signal, window_size):
  
        filtered_signal = signal.copy()
        half_window = window_size // 2
        for i in range(half_window, len(signal) - half_window):
            filtered_signal[i] = np.median(signal[i - half_window:i + half_window])
        return filtered_signal


    @staticmethod
    def low_pass_filter(signal, cutoff_frequency, sampling_rate):
  
        half_sampling_frequency = 0.5 * sampling_rate
        normalized_cutoff_frequency = cutoff_frequency / half_sampling_frequency
        b, a = butter(4, normalized_cutoff_frequency, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal, padtype='constant')
        return filtered_signal
        
    @staticmethod
    def moving_average(signal, window_size):
        smoothed_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        return smoothed_signal
    
    @staticmethod
    def preprocess_data(signal):
        # moving average filtering
        smoothed_signal = Functions.moving_average(signal, window_size=10)
        # median filtering
        smoothed_signal = Functions.median_filter(smoothed_signal, window_size=10)
        # low-pass filtering
        smoothed_signal = Functions.low_pass_filter(smoothed_signal, cutoff_frequency=0.1, sampling_rate=Functions.sampling_rate)
        return smoothed_signal

    @staticmethod
    def plot_spectrogram(time, engine_speed, fs):
            """
            Plot the spectrogram of the engine speed signal.

            Parameters:
            - time: array-like, time data.
            - engine_speed: array-like, engine speed data in U/min.
            - fs: float, sampling frequency in Hz.
            """
            # Convert time and engine speed to numpy arrays
            time = np.array(time)
            engine_speed = np.array(engine_speed)

            # Compute the spectrogram
            f, t, Sxx = spectrogram(engine_speed, fs=fs, nperseg=256, noverlap=128)

            # Plot the spectrogram
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.title('Spectrogram of Engine Speed')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            plt.tight_layout()
            plt.show()
            
    @staticmethod
    def frequency_analysis(signal, sampling_rate, precision=64):
        n = len(signal)
        
        # Sets precision for calculations
        np.set_printoptions(precision=precision)
        # print('1')

        # Converts signal to higher precision
        signal = np.array(signal, dtype=np.float64)
        # print('2')

    

        fft_result = np.fft.fft(signal)
        # print('3')
        d = 1
        freq = np.fft.fftfreq(n, d/sampling_rate)

        # print('3')
        amplitude_spectrum = np.abs(fft_result)
        # print('4')
        # n = 1
        # while(n):
        #     pass

        print("FFT Result:", fft_result)
        print("Frequencies:", freq)
        print("Amplitude Spectrum:", amplitude_spectrum)

        return freq[:n//2], amplitude_spectrum[:n//2]
    
    @staticmethod
    def compute_statistical_features(signal, window_size):
        features = []
        half_window = window_size // 2
        for i in range(half_window, len(signal) - half_window):
            window = signal[i - half_window:i + half_window]
            mean = np.mean(window)
            variance = np.var(window)
            skewness = np.mean(((window - mean) / np.std(window)) ** 3)
            kurtosis = np.mean(((window - mean) / np.std(window)) ** 4) - 3
            features.append([mean, variance, skewness, kurtosis])
        return np.array(features)
    
    @staticmethod
    def plot_predictions(y_test, y_pred):
            plt.figure(figsize=(8, 6))
            
            # Scatter plot for actual values (y_test) in green
            # plt.scatter(y_test, y_test, label='Actual')
            
            # Scatter plot for predicted values (y_pred) in blue
            plt.scatter(y_test, y_pred, color='blue', label='Predicted')
            
            # Plotting the diagonal line
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted')
            plt.grid(True)
            plt.legend()  # Show legend with labels 'Actual' and 'Predicted'
            plt.show()
    
    @staticmethod
    def train_and_evaluate_model(df, target_column):
        # Data Preparation
        X = df.drop(columns=[target_column]).values  # Features
        y = df[target_column].values  # Target variable
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Training model
        model = RandomForestRegressor(n_estimators=100, random_state=30)
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation scores (optional)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)  # Example of 5-fold cross-validation
        
        # Evaluation model
        y_pred = model.predict(X_test_scaled)
        
        # Plot predictions
        Functions.plot_predictions(y_test, y_pred)
        
        # Compute evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print("Cross-Validation Scores:", cv_scores)
        print("Mean absolute error:", mae)
        print("Mean squared error:", mse)
        print("Root mean squared error:", rmse)
        
        return model
    @staticmethod
    def short_time_fourier_transform(signal, sampling_rate, n_fft=512, hop_length=256):
        # Compute Short-Time Fourier Transform (STFT)
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        amplitude = np.abs(stft)
        return amplitude

    # Additional method to compute statistical features over time windows
    @staticmethod
    def compute_statistical_features_over_time(signal, window_size):
        features = []
        for i in range(0, len(signal) - window_size + 1):
            window = signal[i:i + window_size]
            mean = np.mean(window)
            variance = np.var(window)
            skewness = np.mean(((window - mean) / np.std(window)) ** 3)
            kurtosis = np.mean(((window - mean) / np.std(window)) ** 4) - 3
            features.append([mean, variance, skewness, kurtosis])
        return np.array(features)

    # Generating synthetic engine speed data by adding random noise to existing data
    @staticmethod
    def generate_new_engine_speed(engine_speed, noise_level=0.1):
        # Adding random noise to the existing engine speed data
        noise = np.random.normal(loc=0, scale=noise_level, size=len(engine_speed))
        new_engine_speed = engine_speed + noise
        return new_engine_speed
    
    @staticmethod
    def generate_new_engine_load(engine_load, noise_level=0.1):
        # Adding random noise to the existing engine load data
        noise = np.random.normal(loc=0, scale=noise_level, size=len(engine_load))
        new_engine_load = engine_load + noise

        # print(f"predicted_labels{new_engine_load}")
        # while(1): #infinite for - loop to interrupt code implementation
        #     pass

        return new_engine_load

   