import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

def load_mp3(dateipfad):
    try:
        y, sr = librosa.load(dateipfad)
        print("Datei wurde erfolgreich geladen")
        return y, sr
    except Exception as e:
        print(f"Datei konnte nicht geladen werden: {e}")
        return None, None

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def estimate_rpm_from_frequencies(frequencies, amplitudes, cutoff_frequency, rpm_to_frequency_ratio=60):
    motor_frequency_range = (20, cutoff_frequency)  # Hz
    expected_rpm_range = (1000, 6000)  # U/min

    # Filtern der identifizierten Frequenzen im Motorbereich
    filtered_frequencies = [(freq, amp) for freq, amp in zip(frequencies, amplitudes) if motor_frequency_range[0] <= freq <= motor_frequency_range[1]]
    
    if not filtered_frequencies:
        return None

    # Schätzung der Drehzahl basierend auf den identifizierten Frequenzen
    candidate_rpms = []
    for freq, amp in filtered_frequencies:
        candidate_rpm = freq * rpm_to_frequency_ratio
        if expected_rpm_range[0] <= candidate_rpm <= expected_rpm_range[1]:
            candidate_rpms.append(candidate_rpm)

    if not candidate_rpms:
        return None

    # Median-Drehzahl als Schätzung
    estimated_rpm = np.median(candidate_rpms)
    return estimated_rpm

def plot_audio_signal(audio_data, sample_rate, title='Audiosignal im Zeitbereich'):
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title(title)
    plt.xlabel('Zeit (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_fft_with_cutoff(audio_data, sample_rate, start_time, duration, cutoff_frequency, amplitude_threshold):
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    segment = audio_data[start_sample:end_sample]
    
    N = len(segment)
    Y = np.fft.fft(segment)
    frequencies = np.fft.fftfreq(N, d=1/sample_rate)
    
    # Frequenzen oberhalb der cutoff_frequency auf null setzen
    Y[np.abs(frequencies) > cutoff_frequency] = 0

    # Amplituden unterhalb des amplitude_threshold auf null setzen
    Y[np.abs(Y) < amplitude_threshold] = 0

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies[:N//2], np.abs(Y[:N//2]))  # Nur positive Frequenzen
    plt.title(f'Frequenzspektrum des Signals (Zeitraum: {start_time}s bis {start_time + duration}s)')
    plt.xlabel('Frequenz (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    return frequencies, Y

def inverse_transform(Y):
    return np.fft.ifft(Y).real

def estimate_rpm_over_time(audio_data, sample_rate, segment_duration, cutoff_frequency, amplitude_threshold):
    num_segments = int(len(audio_data) / (segment_duration * sample_rate))
    rpms = []

    for i in range(num_segments):
        start_time = i * segment_duration
        frequencies, Y = plot_fft_with_cutoff(audio_data, sample_rate, start_time, segment_duration, cutoff_frequency, amplitude_threshold)
        amplitudes = np.abs(Y)
        rpm = estimate_rpm_from_frequencies(frequencies[:len(frequencies)//2], amplitudes[:len(amplitudes)//2], cutoff_frequency)
        rpms.append(rpm)
        print(f"Segment {i}: Geschätzte Drehzahl = {rpm:.2f} U/min")

    return rpms

# Pfad zur MP3-Datei
dateipfad = "C:\\Users\\User\\Documents\\MCI\\Machinelearing_DataScience\\Project\\Signal-processing\\audio_files\\bmw_short.mp3"

# MP3-Datei laden
audio_data, sample_rate = load_mp3(dateipfad)

if audio_data is not None and sample_rate is not None:
    # Plot des gesamten Audiosignals im Zeitbereich
    plot_audio_signal(audio_data, sample_rate)
    
    # Parameter
    segment_duration = 0.5  # Dauer des Abschnitts in Sekunden
    cutoff_frequency = 1500  # Grenzfrequenz in Hz
    amplitude_threshold = 0.01  # Amplitudenschwellenwert

    # Schätzung der Drehzahl über die Zeit
    rpms = estimate_rpm_over_time(audio_data, sample_rate, segment_duration, cutoff_frequency, amplitude_threshold)
    
    # Plot der geschätzten Drehzahl über die Zeit
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(0, len(rpms) * segment_duration, segment_duration), rpms, marker='o')
    plt.title('Geschätzte Drehzahl über die Zeit')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Drehzahl (U/min)')
    plt.grid(True)
    plt.show()
