import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, find_peaks
import csv

def load_audio(filepath):
    """Load the audio file from the given filepath.

    Args:
        filepath (str): Path to the audio file.

    Returns:
        tuple: Audio time series and sampling rate.
    """
    try:
        y, sr = librosa.load(filepath, sr=None)
        print("File loaded successfully")
        return y, sr
    except Exception as e:
        print(f"File could not be loaded: {e}")
        return None, None

def plot_spectrogram(y, sr, title="Spectrogram"):
    """Plot the spectrogram of the audio signal.

    Args:
        y (array): Audio time series.
        sr (int): Sampling rate of the audio.
        title (str): Title for the spectrogram plot.
    """
    plt.figure(figsize=(10, 6))
    # Compute the Short-Time Fourier Transform (STFT)
    S = librosa.stft(y)
    # Convert the amplitude to decibels
    S_db = librosa.amplitude_to_db(abs(S))
    # Display the spectrogram
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def find_fundamental_frequency(y, sr, nperseg=2048):
    """Find the fundamental frequency of the audio signal using Welch's method.

    Args:
        y (array): Audio time series.
        sr (int): Sampling rate of the audio.
        nperseg (int): Length of each segment for the Welch method.

    Returns:
        float: Fundamental frequency of the audio segment.
    """
    # Compute the power spectral density using Welch's method
    f, Pxx = welch(y, sr, nperseg=nperseg)
    # Find peaks in the power spectral density
    peaks, _ = find_peaks(Pxx, height=np.max(Pxx) * 0.05)  # Peaks with significant amplitude
    if len(peaks) > 0:
        fundamental_frequency = f[peaks[0]]
    else:
        fundamental_frequency = None
    return fundamental_frequency

def calculate_rpm(fundamental_frequency, cylinders=6, strokes=4, correction_factor=0):
    """Calculate the RPM from the fundamental frequency.

    Args:
        fundamental_frequency (float): Fundamental frequency in Hz.
        cylinders (int): Number of cylinders in the engine.
        strokes (int): Number of strokes (2 or 4) in the engine cycle.
        correction_factor (int): Correction factor to adjust the RPM.

    Returns:
        float: Calculated RPM.
    """
    if fundamental_frequency is None:
        return None
    if strokes == 4:
        k = 2  # Each cylinder fires once every two revolutions (for 4-stroke engines)
    elif strokes == 2:
        k = 1  # Each cylinder fires once per revolution (for 2-stroke engines)
    else:
        raise ValueError("Only 2-stroke or 4-stroke engines are supported")
    
    # Calculate RPM: Fundamental frequency is the number of firings per second
    rpm = fundamental_frequency * 60 / (cylinders / k)
    rpm -= correction_factor  # Apply correction factor
    return rpm

def spectral_analysis_and_rpm(filepath, cylinders=6, strokes=4, correction_factor=200, csv_filepath="rpm_data.csv"):
    """Perform spectral analysis and RPM estimation from the audio file.

    Args:
        filepath (str): Path to the audio file.
        cylinders (int): Number of cylinders in the engine.
        strokes (int): Number of strokes (2 or 4) in the engine cycle.
        correction_factor (int): Correction factor to adjust the RPM.
        csv_filepath (str): Path to save the RPM data CSV file.
    """
    # Load the audio file
    y, sr = load_audio(filepath)
    if y is None or sr is None:
        return

    # Plot the spectrogram of the entire audio signal
    plot_spectrogram(y, sr, title="Spectrogram of the Audio Signal")

    seg_duration = 0.5  # Length of the time segment in seconds
    seg_samples = int(seg_duration * sr)  # Convert segment duration to samples
    rpms = []

    # Analyze each segment of the audio
    for start in range(0, len(y), seg_samples):
        end = start + seg_samples
        if end > len(y):
            break
        segment = y[start:end]
        # Find the fundamental frequency of the segment
        fundamental_frequency = find_fundamental_frequency(segment, sr)
        # Calculate the RPM from the fundamental frequency
        rpm = calculate_rpm(fundamental_frequency, cylinders, strokes, correction_factor)
        rpms.append(rpm)

    # Write RPM data to CSV
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "RPM"])
        for i, rpm in enumerate(rpms):
            writer.writerow([i * seg_duration, rpm])
    
    # Plot the estimated RPM over time
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(rpms)) * seg_duration, rpms, marker='o')
    plt.title('Estimated RPM over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('RPM')
    plt.grid(True)
    plt.show()

# Path to the MP3 file
filepath = "C:\\Users\\User\\Documents\\MCI\\Machinelearing_DataScience\\Project\\Signal-processing\\audio_files\\bmw_short.mp3"

# Perform spectral analysis and RPM estimation
spectral_analysis_and_rpm(filepath, cylinders=6, strokes=4, correction_factor=200, csv_filepath="rpm_data.csv")
