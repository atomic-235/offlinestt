#!/usr/bin/env python3

import os
import os.path as osp
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
    Transcribe Russian audio file using Whisper model, handling long files by processing in chunks
    """
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Get audio info first
        info = sf.info(audio_path)
        sample_rate = info.samplerate
        duration = info.duration
        print(f"Audio duration: {duration:.2f} seconds")

        # Load Whisper model for Russian once
        print("Loading Whisper model for Russian...")
        model_name = "openai/whisper-base"
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        processor = WhisperProcessor.from_pretrained(model_name)

        # Create a temporary directory for chunks
        temp_dir = osp.join(osp.dirname(audio_path), "temp_chunks")
        os.makedirs(temp_dir, exist_ok=True)

        # Split audio into 30-second chunks
        chunk_duration = 30  # seconds
        num_chunks = int(np.ceil(duration / chunk_duration))

        all_transcriptions = []

        print(f"Splitting into {num_chunks} chunks of {chunk_duration} seconds each")

        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min(start_time + chunk_duration, duration)

            # Progress indicator
            if i % max(1, num_chunks // 10) == 0:
                print(
                    f"Processing chunk {i + 1}/{num_chunks} ({start_time:.1f}s to {end_time:.1f}s)"
                )

            # Load the chunk
            try:
                chunk_waveform, _ = sf.read(
                    audio_path,
                    start=int(start_time * sample_rate),
                    stop=int(end_time * sample_rate),
                    dtype="float32",
                )

                # Convert to mono if needed
                if len(chunk_waveform.shape) > 1:
                    chunk_waveform = np.mean(chunk_waveform, axis=1)

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    duration_chunk = len(chunk_waveform) / sample_rate
                    new_length = int(duration_chunk * 16000)
                    old_indices = np.arange(len(chunk_waveform))
                    new_indices = np.linspace(0, len(chunk_waveform) - 1, new_length)
                    chunk_waveform = np.interp(new_indices, old_indices, chunk_waveform)
                    current_sample_rate = 16000
                else:
                    current_sample_rate = sample_rate

                # Process this chunk
                inputs = processor(
                    chunk_waveform,
                    sampling_rate=current_sample_rate,
                    return_tensors="pt",
                ).to(device)

                # Generate transcription
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
                transcription = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                all_transcriptions.append(transcription)

                # Clean up to save memory
                del inputs, generated_ids
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing chunk {i + 1}: {e}")
                all_transcriptions.append(f"[Error in chunk {i + 1}]")

        # Clean up temporary directory
        import shutil

        if osp.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # Join all transcriptions with spaces
        return " ".join(all_transcriptions)

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
