#!/usr/bin/env python3

import os
import sys
import warnings

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Suppress warnings and verbose output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")


def transcribe_audio(audio_path):
    """
    Transcribe Russian audio file using Whisper model
    """
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Load audio file
        waveform, sample_rate = sf.read(audio_path, dtype="float32")

        # Convert to mono if needed
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)

        # Resample to 16kHz which is required for Whisper
        if sample_rate != 16000:
            duration = len(waveform) / sample_rate
            new_length = int(duration * 16000)
            old_indices = np.arange(len(waveform))
            new_indices = np.linspace(0, len(waveform) - 1, new_length)
            waveform = np.interp(new_indices, old_indices, waveform)
            sample_rate = 16000
            print(f"Resampled audio to 16kHz")

        # Load Whisper model for Russian
        print("Loading Whisper model for Russian...")
        model_name = "openai/whisper-base"
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        processor = WhisperProcessor.from_pretrained(model_name)

        # Process audio with forced Russian language
        print("Processing audio...")
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").to(
            device
        )

        # Generate transcription with Russian language specified
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="ru", task="transcribe"
        )

        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=200,
            )

        # Decode the generated ids to text
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return transcription

    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_russian.py <audio_file_path>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    # Transcribe the audio
    transcription = transcribe_audio(audio_file)

    if transcription:
        print("\nTranscription (Russian):")
        print(transcription)
    else:
        print("Transcription failed.")
        sys.exit(1)
