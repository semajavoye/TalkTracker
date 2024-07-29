import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pyaudio
import wave
import os

class SpeakerIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speaker Identification")
        
        self.label_frame = tk.Frame(root)
        self.label_frame.pack(pady=10)

        self.label_list = []
        self.file_list = []
        
        self.label_label = tk.Label(self.label_frame, text="Label:")
        self.label_label.pack(side=tk.LEFT)
        self.label_entry = tk.Entry(self.label_frame)
        self.label_entry.pack(side=tk.LEFT)
        
        self.add_file_button = tk.Button(root, text="Add MP3 File", command=self.add_file)
        self.add_file_button.pack(pady=5)
        
        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=5)
        
        self.predict_file_button = tk.Button(root, text="Predict from File", command=self.predict_from_file)
        self.predict_file_button.pack(pady=5)
        
        self.predict_mic_button = tk.Button(root, text="Predict from Microphone", command=self.predict_from_mic)
        self.predict_mic_button.pack(pady=5)
        
        self.features = []
        self.labels = []
        self.model = None
        self.le = None
    
    def add_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
        if file_path:
            label = self.label_entry.get().strip()
            if label:
                self.file_list.append(file_path)
                self.label_list.append(label)
                self.label_entry.delete(0, tk.END)
                messagebox.showinfo("Success", f"File {file_path} added with label {label}")
            else:
                messagebox.showwarning("Warning", "Please enter a label before adding a file")
    
    def convert_mp3_to_wav(self, mp3_path, wav_path):
        y, sr = librosa.load(mp3_path, sr=None)
        sf.write(wav_path, y, sr)
    
    def extract_mfcc(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    
    def train_model(self):
        if len(set(self.label_list)) < 2:
            messagebox.showwarning("Warning", "Need at least two different labels to train the model")
            return

        if not self.file_list:
            messagebox.showwarning("Warning", "No files to train on")
            return

        wav_files = []
        for mp3_file in self.file_list:
            wav_file = mp3_file.replace(".mp3", ".wav")
            self.convert_mp3_to_wav(mp3_file, wav_file)
            wav_files.append(wav_file)

        self.features = [self.extract_mfcc(wav) for wav in wav_files]
        self.labels = self.label_list

        # Convert features and labels to numpy arrays for easier manipulation
        X = np.array(self.features)
        y = np.array(self.labels)

        # Check if any class has fewer than 2 samples
        unique_labels, label_counts = np.unique(y, return_counts=True)
        if any(count < 2 for count in label_counts):
            messagebox.showwarning("Warning", "Some classes have fewer than 2 samples. Please provide more data.")
            return

        try:
            # Attempt stratified split
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            split = sss.split(X, y)
            train_index, test_index = next(split)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        except ValueError as e:
            # If stratified split fails, fallback to regular train_test_split
            print("Stratified split failed, falling back to regular split:", e)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.le = LabelEncoder()
        self.le.fit(self.labels)
        y_train = self.le.transform(y_train)
        y_test = self.le.transform(y_test)

        # Print label distributions after splitting
        print("Training labels distribution:")
        unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique_train_labels, train_counts):
            print(f"{self.le.inverse_transform([label])[0]}: {count}")

        print("Test labels distribution:")
        unique_test_labels, test_counts = np.unique(y_test, return_counts=True)
        for label, count in zip(unique_test_labels, test_counts):
            print(f"{self.le.inverse_transform([label])[0]}: {count}")

        if len(unique_train_labels) < 2 or len(unique_test_labels) < 2:
            messagebox.showwarning("Warning", "The split resulted in fewer than 2 classes in either the training or testing set")
            return

        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_train, y_train)

        accuracy = self.model.score(X_test, y_test)
        messagebox.showinfo("Model Trained", f"Model trained with accuracy: {accuracy}")


    def predict_from_file(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        file_path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3"), ("WAV files", "*.wav")])
        if file_path:
            if file_path.endswith(".mp3"):
                wav_file = file_path.replace(".mp3", ".wav")
                self.convert_mp3_to_wav(file_path, wav_file)
                file_path = wav_file
            
            features = self.extract_mfcc(file_path)
            prediction = self.model.predict([features])
            speaker = self.le.inverse_transform(prediction)
            messagebox.showinfo("Prediction", f"Predicted Speaker: {speaker[0]}")
    
    def predict_from_mic(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "mic_recording.wav"
        
        p = pyaudio.PyAudio()
        
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        
        messagebox.showinfo("Recording", "Recording for 5 seconds")
        
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        features = self.extract_mfcc(WAVE_OUTPUT_FILENAME)
        prediction = self.model.predict([features])
        speaker = self.le.inverse_transform(prediction)
        messagebox.showinfo("Prediction", f"Predicted Speaker: {speaker[0]}")
        
        os.remove(WAVE_OUTPUT_FILENAME)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeakerIDApp(root)
    root.mainloop()
