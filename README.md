# 🎤 Real-Time Voice Activity Detection (VAD) using Silero and Microphone Input

This Python script performs **real-time voice activity detection (VAD)** from your system's microphone using the [Silero VAD](https://github.com/snakers4/silero-vad) model. It prints "Speech Detected" or "Silence" on your terminal for each short audio segment captured.

---

## 📦 Dependencies

Install the required Python packages:

```bash
pip install torch sounddevice scipy numpy
```

### ✅ Description of Dependencies

| Package | Purpose |
|--------|---------|
| `torch` | Runs the Silero VAD model |
| `sounddevice` | Streams real-time microphone input |
| `numpy` | Handles audio data in array format |
| `scipy` | (optional for future use like resampling) |

---

## ▶️ How to Run

1. Make sure your system microphone is enabled.
2. Run the script:

```bash
python vad_mic_stream.py
```

You’ll see terminal output like:

```
🎙️ Listening... Press Ctrl+C to stop.
🔴 Silence
🟢 Speech detected at [{'start': 0.0, 'end': 0.7}]
🔴 Silence
...
```

---

## 🧠 Understanding the Code

### Constants

```python
SAMPLE_RATE = 16000  # Target sample rate for Silero VAD
BLOCK_DURATION = 0.7  # Length of each audio chunk in seconds
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)  # Samples per chunk
```

- **`16000 Hz`** is the required sampling rate for the Silero model.
- **`0.7 seconds`** of audio is processed at a time (adjustable).
- The input is flattened and converted to a PyTorch tensor.

---

### Input Stream Configuration

```python
stream = sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE)
```

- `channels=1`: Mono audio.
- `blocksize=BLOCK_SIZE`: Amount of data per callback.
- `callback`: Puts audio blocks in a queue to be processed.

---

### Core Logic

```python
timestamps = get_speech_timestamps(block_tensor, model, return_seconds=True)
```

- Uses Silero's `get_speech_timestamps()` to detect whether speech occurred in the chunk.
- `return_seconds=True` returns `start` and `end` timestamps in seconds.

---

## ✏️ Customization

- **Change chunk duration:** Modify `BLOCK_DURATION`.
- **Use in noisy environments:** You may use noise suppression or tweak Silero thresholds (see advanced examples in [Silero repo](https://github.com/snakers4/silero-vad)).
- **Log to a file:** Replace the `print()` statement with logging.
- **Save speech segments:** Use `collect_chunks()` and `save_audio()` from the model's utils.

---

## ❓ FAQ

### Does this differentiate between speakers?
No — this only detects **presence of speech**. It does not support speaker identification. For that, consider [pyannote-audio](https://github.com/pyannote/pyannote-audio) or [WhisperX](https://github.com/m-bain/whisperx).

### Can I use stereo audio?
No — Silero expects mono, 16kHz input. If you have stereo, downmix before use.

### Is this real-time?
Yes — it processes chunks as they are captured from the microphone.

---

## 📜 Credits

- [Silero VAD](https://github.com/snakers4/silero-vad) by Snakers4
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- [PyTorch](https://pytorch.org/)

---

## 🛑 Stop the script
Just hit `Ctrl+C` to stop microphone listening safely.
