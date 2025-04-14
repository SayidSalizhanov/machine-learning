import librosa
import matplotlib.pyplot as plt

def main():
    audio_path = "C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\Ledy GaGa - Judas.mp3"
    y, sr = librosa.load(audio_path)
    plt.plot(range(len(y)), y)
    plt.show()

if __name__ == "__main__":
    main()