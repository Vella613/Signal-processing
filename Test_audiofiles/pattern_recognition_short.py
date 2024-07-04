import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, find_peaks
import csv

def load_audio(filepath):
    """Load the audio file from the given filepath."""
    try:
        y, sr = librosa.load(filepath, sr=None)
        print("File loaded successfully")
        return y, sr
    except Exception as e:
        print(f"File could not be loaded: {e}")
        return None, None

def plot_spectrogram(y, sr, title="Spectrogram"):
    """Plot the spectrogram of the audio signal."""
    plt.figure(figsize=(10, 6))
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def find_fundamental_frequency(y, sr, nperseg=2048):
    """Find the fundamental frequency of the audio signal using Welch's method."""
    f, Pxx = welch(y, sr, nperseg=nperseg)
    peaks, _ = find_peaks(Pxx, height=np.max(Pxx) * 0.05)  # Fundamental frequencies with significant amplitude
    if len(peaks) > 0:
        fundamental_frequency = f[peaks[0]]
    else:
        fundamental_frequency = None
    return fundamental_frequency

def calculate_rpm(fundamental_frequency, cylinders=6, strokes=4, correction_factor=0):
    """Calculate the RPM from the fundamental frequency."""
    if fundamental_frequency is None:
        return None
    if strokes == 4:
        k = 2  # Each cylinder fires once every two revolutions (for 4-stroke engines)
    elif strokes == 2:
        k = 1  # Each cylinder fires once per revolution (for 2-stroke engines)
    else:
        raise ValueError("Only 2-stroke or 4-stroke engines are supported")
    
    # The fundamental frequency is the number of firings per second. To calculate RPM:
    rpm = fundamental_frequency * 60 / (cylinders / k)
    rpm -= correction_factor  # Apply correction factor
    return rpm

def spectral_analysis_and_rpm(filepath, cylinders=6, strokes=4, correction_factor=200, csv_filepath="rpm_data.csv"):
    """Perform spectral analysis and RPM estimation from the audio file."""
    y, sr = load_audio(filepath)
    if y is None or sr is None:
        return

    plot_spectrogram(y, sr, title="Spectrogram of the Audio Signal")

    seg_duration = 0.5  # Length of the time segment in seconds
    seg_samples = int(seg_duration * sr)
    rpms = []

    for start in range(0, len(y), seg_samples):
        end = start + seg_samples
        if end > len(y):
            break
        segment = y[start:end]
        fundamental_frequency = find_fundamental_frequency(segment, sr)
        rpm = calculate_rpm(fundamental_frequency, cylinders, strokes, correction_factor)
        rpms.append(rpm)

    # Write RPM data to CSV
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "RPM"])
        for i, rpm in enumerate(rpms):
            writer.writerow([i * seg_duration, rpm])
    
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
