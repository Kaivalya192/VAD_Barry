# pip install torch sounddevice scipy numpy

import torch
import sounddevice as sd
import queue
import time

# Load Silero VAD model and utilities
torch.set_num_threads(1)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Constants
SAMPLE_RATE = 16000  # Silero expects 16kHz # Mono expects 8kHz
BLOCK_DURATION = 0.7  # seconds
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ {status}")
    q.put(indata.copy())

# Start input stream
stream = sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE)
stream.start()

print("🎙️ Listening... Press Ctrl+C to stop.")

try:
    while True:
        block = q.get()
        block = block.flatten()
        block_tensor = torch.from_numpy(block).float()
        # Apply VAD
        timestamps = get_speech_timestamps(block_tensor, model, return_seconds=True)
        if timestamps:
            print(f"🟢 Speech detected at {timestamps}")
        else:
            print("🔴 Silence")
        time.sleep(BLOCK_DURATION)
except KeyboardInterrupt:
    print("\n🛑 Stopped.")
finally:
    stream.stop()
    stream.close()
