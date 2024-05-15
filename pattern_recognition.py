import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load_mp3(dateipfad):
    try:
        y, sr = librosa.load(dateipfad)
        print("Datei wurde erfolgreich geladen")
        return y, sr
        
    except Exception as e:
        print(f"Datei konnte nicht geladen werden: {e}")
        return None, None

def estimate_rpm_from_frequencies(frequencies):
    # Annahme: Motorspezifische Informationen
    rpm_to_frequency_ratio = 60  # Eine Umdrehung pro Sekunde entspricht 60 Hz
    motor_frequency_range = (100, 10000)  # Hz
    expected_rpm_range = (1000, 6000)  # U/min
    
    # Filtern der identifizierten Frequenzen auf den Motorbereich
    filtered_frequencies = [freq for freq in frequencies if motor_frequency_range[0] <= freq <= motor_frequency_range[1]]
    
    # Schätzung der Drehzahl basierend auf den identifizierten Frequenzen
    candidate_rpms = []
    for freq in filtered_frequencies:
        candidate_rpm = freq / rpm_to_frequency_ratio
        if expected_rpm_range[0] <= candidate_rpm <= expected_rpm_range[1]:
            candidate_rpms.append(candidate_rpm)
    
    # Falls keine gültige Drehzahl gefunden wird, geben wir None zurück
    if not candidate_rpms:
        return None
    
    # Wir nehmen die Median-Drehzahl als Schätzung
    estimated_rpm = np.median(candidate_rpms)
    return estimated_rpm


# Pfad zur MP3-Datei
dateipfad = "C:\\Users\\User\\Documents\\MCI\\Machinelearing_DataScience\\Project\\Signal-processing\\audio_files\\bmw_short.mp3"

# MP3-Datei laden
audio_data, sample_rate = load_mp3(dateipfad)

if audio_data is not None and sample_rate is not None:
    # Fourier-Transformation des Signals
    
    # Ändere die Länge des Signals auf das Vierfache
    N = len(audio_data) * 4
    Y = np.fft.fft(audio_data, n=N)


    # Länge des Signals
    

    # Berechnung der Frequenzen
    frequencies = np.fft.fftfreq(N, d=1/sample_rate)

    # Plot des Spektrums
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies[:N//2], np.abs(Y[:N//2]))  # Nur positive Frequenzen
    plt.title('Frequenzspektrum des Signals')
    plt.xlabel('Frequenz (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Schätzung der Drehzahl
    estimated_rpm = estimate_rpm_from_frequencies(frequencies)
    if estimated_rpm is not None:
        print("Geschätzte Drehzahl:", estimated_rpm)
    else:
        print("Keine Frequenzen im Motorbereich gefunden.")
