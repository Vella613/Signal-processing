import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from class_functions import Functions
from Regression_class import LinearRegression

def  main():
    data = pd.read_excel("engine2_normalized.xlsx")
    data = data.interpolate(method='linear')


    data = data.dropna(subset=['engine_speed', 'engine_load']) 


    engine_speed = data['engine_speed']
    engine_load = data['engine_load']

    engine_speed_freq, engine_speed_spectrum = Functions.frequency_analysis(engine_speed, Functions.sampling_rate)
    engine_load_freq, engine_load_spectrum = Functions.frequency_analysis(engine_load, Functions.sampling_rate)


    plt.figure(figsize=(10, 6))
    plt.plot(engine_speed, label='Engine Speed')
    plt.plot(engine_load, label='Engine Load')
    plt.title('Engine Signals')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    engine_speed_normalized = (engine_speed - engine_speed.mean()) / engine_speed.std()
    engine_load_normalized = (engine_load - engine_load.mean()) / engine_load.std()

    # Applys logarithmic transformation to emphasize small variations
    engine_speed_log = np.log(engine_speed_normalized.abs() + 1)
    engine_load_log = np.log(engine_load_normalized.abs() + 1)


    # moving average filtering
    engine_speed_smoothed_ma = Functions.moving_average(engine_speed, window_size=10)
    engine_load_smoothed_ma = Functions.moving_average(engine_load, window_size=10)

    #  median filtering
    engine_speed_smoothed_median = Functions.median_filter(engine_speed, window_size=10)
    engine_load_smoothed_median = Functions.median_filter(engine_load, window_size=10)

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

    plt.subplot(2, 1, 2)
    plt.plot(engine_load, label='Original Engine Load', alpha=0.5)
    plt.plot(engine_load_smoothed_ma, label='Moving Average Filtered', linestyle='--', color='red')
    plt.plot(engine_load_smoothed_median, label='Median Filtered', linestyle='--', color='green')
    plt.title('Engine Load Signals with Moving Average and Median Filtering')
    plt.xlabel('Time')
    plt.ylabel('Engine Load')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculating mean engine speed and engine load after preprocessing
    engine_speed_mean = engine_speed.mean()
    engine_load_mean = engine_load.mean()

    print("Mean Engine Speed:", engine_speed_mean)
    print("Mean Engine Load:", engine_load_mean)

    # Frequency analysis for engine speed
    engine_speed_freq, engine_speed_spectrum = Functions.frequency_analysis(engine_speed, Functions.sampling_rate)

    # Frequency analysis for engine load
    engine_load_freq, engine_load_spectrum = Functions.frequency_analysis(engine_load, Functions.sampling_rate)

    # Plot frequency spectra
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(engine_speed_freq, engine_speed_spectrum)
    plt.title('Engine Speed Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(engine_load_freq, engine_load_spectrum)
    plt.title('Engine Load Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Engine Speed after Preprocessing:", engine_speed)
    print("Engine Load after Preprocessing:", engine_load)

    print("Engine_speed after preprocessing:", engine_speed)
    print("Engine_load after preprocessing:", engine_load)



    # Training  model
  
    # model = Functions.train_and_evaluate_model(data, target_column)

    #predicting new data
    # new_data = # Load new signal data
    # predicted_labels = Functions.predict_new_data(model, new_data)


    print("Original engine speed:", engine_speed[:10])  

    # adjusts noise_level parameter as needed
    existing_engine_speed = data['engine_speed']  
    existing_engine_load = data['engine_load']  

    # trains and evaluates the model for engine speed
    model_engine_speed = Functions.train_and_evaluate_model(data, 'engine_speed')  

    # trains and evaluates the model for engine load
    model_engine_load = Functions.train_and_evaluate_model(data, 'engine_load')  

    # generates synthetic engine speed and engine load data
    new_engine_speed = Functions.generate_new_engine_speed(existing_engine_speed)
    new_engine_load = Functions.generate_new_engine_load(existing_engine_load)

    # creates a DataFrame for the new synthetic data
    new_data = pd.DataFrame({
        'engine_speed': new_engine_speed,
        'engine_load': new_engine_load

    })

    # uses trained models to predict engine speed and engine load for the new data  FALSE NOT WORKING - TO BE REMOVED
    # predicted_engine_speed =Functions.call_predcited_data(model_engine_speed, new_data)
    # predicted_engine_load =Functions.call_predcited_data(model_engine_load, new_data)

    # print("Predicted engine speed for new data:", predicted_engine_speed)
    # print("Predicted engine load for new data:", predicted_engine_load)

    # Plots original and generated engine speed
    plt.figure(figsize=(10, 6))
    plt.plot(existing_engine_speed, label='Original Engine Speed', color='blue')
    plt.plot(new_engine_speed, label='Generated Engine Speed', color='yellow')
    plt.title('Original and Generated Engine Speed')
    plt.xlabel('Time')
    plt.ylabel('Engine Speed')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plots original and generated engine load
    plt.figure(figsize=(10, 6))
    plt.plot(existing_engine_load, label='Original Engine Load', color='blue')
    plt.plot(new_engine_load, label='Generated Engine Load', color='red')
    plt.title('Original and Generated Engine Load')
    plt.xlabel('Time')
    plt.ylabel('Engine Load')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()