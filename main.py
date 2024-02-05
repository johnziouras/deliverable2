import whisper
import pyttsx3
import sounddevice as sd
from scipy.io.wavfile import write
import gtts
from playsound import playsound

def record_and_save_wav(filename, duration=4, sample_rate=44100):
    freq = 44100

    # Recording duration
    duration = 4

    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=1)

    # Record audio for the given number of seconds
    sd.wait()

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write(filename, freq, recording)


def process_wav(input_f_name, output_f_name, model):
    print('Processing...')
    result = model.transcribe(input_f_name, fp16=False)
    print(result['text'])
    tts = gtts.gTTS(result['text'])
    tts.save(output_f_name)
    playsound(output_f_name)


def main():
    record_name = "audio/record.wav"
    output_name = "audio/output.wav"

    model = whisper.load_model('base')  # Initialize whisper model

    print('Press x to exit the program')
    while True:
        user_input = input('Press c to begin recording\n')
        if user_input.lower() == 'c':
            record_and_save_wav(filename=record_name)
            process_wav(input_f_name=record_name, output_f_name=output_name, model=model)
        if user_input.lower() == 'x':
            break


if __name__ == "__main__":
    main()


