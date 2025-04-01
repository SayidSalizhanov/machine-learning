import librosa
import numpy as np

def main():
    audio_path = "C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\Ledy GaGa - Judas.mp3"
    y, sr = librosa.load(audio_path)

if __name__ == "__main__":
    main()