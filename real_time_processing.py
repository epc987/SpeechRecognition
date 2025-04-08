import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import time
import pyaudio
import wave


model = tf.keras.models.load_model("model.h5")

# Convert the model to TensorFlow Lite

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TensorFlow Lite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite.")


SAMPLE_RATE = 16000 # Common sample rate for speech models
DURATION = 2 # Length of audio snippet

def record_audio(duration, sample_rate):
    print("Listening...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait() # Wait for recording to complete
    return np.squeeze(audio) # Flatten the audio array

def preprocess_audio(audio, sample_rate):
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_fft=512, hop_length=256, n_mels=40
    )
    # Convert to logarithmic scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # Add batch channel dimensions
    return np.expand_dims(log_mel_spectrogram, axis=0).astype(np.float32)


interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

#Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_inference(audio):
    processed_audio = preprocess_audio(audio, SAMPLE_RATE)
    interpreter.set_tensor(input_details[0]['index'], processed_audio)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def measure_metrics(audio):
    start_time = time.time()
    output = run_inference(audio)
    inference_time = time.time() - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")
    return output, inference_time

try:
    while True:
        audio = record_audio(DURATION, SAMPLE_RATE) 
        
        #Measure metrics and get model prediction
        output, inference_time = measure_metrics(audio)

        # Print predicted class
        predicted_class = np.argmax(output)
        print(f"Predicted Command: {predicted_class}, Inference Time: {inference_time:.4f} seconds")

except KeyboardInterrupt:
    print("Stopping real-time processing")


