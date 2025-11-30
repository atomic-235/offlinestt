import os

from faster_whisper import WhisperModel
from tqdm import tqdm

model_size = "base"  # Using smaller base model for faster processing

# Use CPU with INT8 for faster processing
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Get input and output paths
audio_path = "/home/antonym/Downloads/Recording 2.mp3"
output_path = "/home/antonym/projects/offlinestt/transcription.md"

# Create output file and write header
with open(output_path, "w") as md_file:
    md_file.write("# Transcription Results\n\n")
    md_file.write(f"## Audio File: {os.path.basename(audio_path)}\n\n")
    md_file.write("## Transcription:\n\n")

print("Starting transcription...")
segments, info = model.transcribe(audio_path, beam_size=5, language="ru")


# Process and save each segment on the fly with progress bar
with open(output_path, "a") as md_file:
    with tqdm(desc="Transcribing", unit=" segment") as pbar:
        for i, segment in enumerate(segments):
            # Print to console
            print(segment.text)

            # Write to markdown file immediately
            md_file.write(f" {segment.text}\n")

            # Add a line break in markdown for readability
            if i % 5 == 4:  # Every 5 sentences, add an extra break
                md_file.write("\n")

            # Update progress bar
            pbar.update(1)
            pbar.set_description(f"Segment {i + 1}")

print(f"Transcription complete! Results saved to {output_path}")
