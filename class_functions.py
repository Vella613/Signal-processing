import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# from pydub import AudioSegment
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from Regression_class import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import io

class Functions:
    sampling_rate = 1  # Hz

    
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
        freq = np.fft.fftfreq(n, d /sampling_rate)

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
    def plot_predictions(y_test, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.grid(True)
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
        
        # training model
        model = RandomForestRegressor(n_estimators=100, random_state=30)
        model.fit(X_train_scaled, y_train)
        
        # evaluation model
        y_pred = model.predict(X_test_scaled)

        # print(f'y_pred : {y_pred}')
        # while(1):
        #     pass

        Functions.plot_predictions(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print("Mean absolute error:", mae)
        print("Mean squared error:", mse)
        print("Root mean squared error:", rmse)
        
    
        
        return model

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

  

 
