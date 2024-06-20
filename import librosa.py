import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, find_peaks

def lade_audio(dateipfad):
    try:
        y, sr = librosa.load(dateipfad, sr=None)
        print("Datei wurde erfolgreich geladen")
        return y, sr
    except Exception as e:
        print(f"Datei konnte nicht geladen werden: {e}")
        return None, None

def plot_spektrogramm(y, sr, titel="Spektrogramm"):
    plt.figure(figsize=(10, 6))
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(titel)
    plt.show()

def finde_grundfrequenz(y, sr, nperseg=1024):
    f, Pxx = welch(y, sr, nperseg=nperseg)
    peaks, _ = find_peaks(Pxx, height=np.max(Pxx) * 0.1)  # Grundfrequenzen mit signifikanter Amplitude
    if len(peaks) > 0:
        grundfrequenz = f[peaks[0]]
    else:
        grundfrequenz = None
    return grundfrequenz

def berechne_drehzahl(grundfrequenz, zylinder=6, takte=4):
    if grundfrequenz is None:
        return None
    if takte == 4:
        k = 2  # Jeder Zylinder zündet einmal pro zwei Umdrehungen (bei 4-Takt-Motoren)
    elif takte == 2:
        k = 1  # Jeder Zylinder zündet einmal pro Umdrehung (bei 2-Takt-Motoren)
    else:
        raise ValueError("Nur 2-Takt oder 4-Takt-Motoren werden unterstützt")
    
    # Die Grundfrequenz ist die Anzahl der Zündungen pro Sekunde. Um die Drehzahl zu berechnen:
    drehzahl = grundfrequenz * 60 / (zylinder / k)
    return drehzahl

def spektralanalyse_und_drehzahl(dateipfad, zylinder=4, takte=4):
    y, sr = lade_audio(dateipfad)
    if y is None or sr is None:
        return

    plot_spektrogramm(y, sr, titel="Spektrogramm des Audiosignals")

    seg_dauer = 0.5  # Länge des Zeitabschnitts in Sekunden
    seg_samples = int(seg_dauer * sr)
    drehzahlen = []

    for start in range(0, len(y), seg_samples):
        ende = start + seg_samples
        if ende > len(y):
            break
        segment = y[start:ende]
        grundfrequenz = finde_grundfrequenz(segment, sr)
        drehzahl = berechne_drehzahl(grundfrequenz, zylinder, takte)
        drehzahlen.append(drehzahl)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(drehzahlen)) * seg_dauer, drehzahlen, marker='o')
    plt.title('Geschätzte Drehzahl über die Zeit')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Drehzahl (U/min)')
    plt.grid(True)
    plt.show()

# Pfad zur MP3-Datei
dateipfad = "C:\\Users\\User\\Documents\\MCI\\Machinelearing_DataScience\\Project\\Signal-processing\\audio_files\\bmw_short.mp3"

# Durchführung der Spektralanalyse und Drehzahlschätzung
spektralanalyse_und_drehzahl(dateipfad, zylinder=4, takte=4)