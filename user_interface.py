import streamlit as st
import sqlite3
import librosa
import numpy as np
import os

# Initialize SQLite database
def init_db():
    db = sqlite3.connect('engine_speed.db')
    cursor = db.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS engine_speeds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            recognized INTEGER DEFAULT 0
        )
    ''')
    db.commit()
    db.close()

# Function to insert data into SQLite database
def insert_data_to_db(filename, filepath):
    try:
        db = sqlite3.connect('engine_speed.db')
        cursor = db.cursor()
        cursor.execute("INSERT INTO engine_speeds (filename, filepath, recognized) VALUES (?, ?, 0)", (filename, filepath,))
        db.commit()
        db.close()
    except sqlite3.Error as e:
        print(f"Error inserting data into database: {e}")

# Function to fetch all filenames from database
def fetch_filenames_from_db():
    return ['jet_engine.mp3', 'bmw_short.mp3']  # Directly return the known MP3 filenames

# Function to update recognition status in database
def update_recognition_status(filename):
    try:
        db = sqlite3.connect('engine_speed.db')
        cursor = db.cursor()
        cursor.execute("UPDATE engine_speeds SET recognized = 1 WHERE filename = ?", (filename,))
        db.commit()
        db.close()
    except sqlite3.Error as e:
        print(f"Error updating recognition status: {e}")

# Function to display recognition status
def display_recognition_status():
    st.subheader("Recognition Status")
    filenames = fetch_filenames_from_db()
    if len(filenames) == 0:
        st.write("No files found.")
        return
    
    selected_filename = st.selectbox("Select a file", filenames)
    if st.button("Update Recognition Status"):
        update_recognition_status(selected_filename)
        st.success(f"Recognition status updated for {selected_filename}")

# Function to handle file uploads
def handle_file_upload():
    uploaded_file = st.file_uploader("Upload an engine speed file", type=["mp3", "wav"])
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        insert_data_to_db(uploaded_file.name, file_path)
        st.success("File uploaded and data saved to database.")

# Initialize SQLite database
def init_db():
    db = sqlite3.connect('engine_speed.db')
    cursor = db.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            mfcc TEXT NOT NULL,
            chroma TEXT NOT NULL,
            spectral_contrast TEXT NOT NULL
        )
    ''')
    db.commit()
    db.close()

# Function to extract and save audio features
def extract_and_save_features(mp3_path):
    try:
        # Load audio file
        y, sr = librosa.load(mp3_path)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Convert features to JSON strings for storage in SQLite TEXT fields
        mfcc_str = np.array2string(mfccs, separator=',')
        chroma_str = np.array2string(chroma, separator=',')
        spectral_contrast_str = np.array2string(spectral_contrast, separator=',')

        # Save to database
        db = sqlite3.connect('engine_speed.db')
        cursor = db.cursor()
        cursor.execute("INSERT INTO audio_features (filename, filepath, mfcc, chroma, spectral_contrast) VALUES (?, ?, ?, ?, ?)", 
                    (os.path.basename(mp3_path), mp3_path, mfcc_str, chroma_str, spectral_contrast_str))
        db.commit()
        db.close()

        print(f"Features extracted and saved for {mp3_path}")
    except Exception as e:
        print(f"Error processing {mp3_path}: {e}")

# Main Streamlit application code
def main():
    st.title("Engine Speed Recognition")
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose an option", ["Upload Engine Speed", "Recognition Status"])

    if app_mode == "Upload Engine Speed":
        handle_file_upload()
    elif app_mode == "Recognition Status":
        display_recognition_status()

# Ensure app runs only if called directly
if __name__ == '__main__':
    init_db()
    mp3_files = ['bmw_short.mp3',
        'jet_engine.mp3']
    extract_and_save_features(mp3_files)
    main()
