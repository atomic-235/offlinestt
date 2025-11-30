import os

from faster_whisper import WhisperModel

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

print(
    "Detected language '%s' with probability %f"
    % (info.language, info.language_probability)
)

# Write language info to file
with open(output_path, "a") as md_file:
    md_file.write(
        f"**Detected Language:** {info.language} (probability: {info.language_probability:.2f})\n\n"
    )
    md_file.write("---\n\n")

# Process and save each segment on the fly
with open(output_path, "a") as md_file:
    for i, segment in enumerate(segments):
        # Print to console
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        # Write to markdown file immediately
        md_file.write(
            f"{i + 1}. [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
        )

        # Add a line break in markdown for readability
        if i % 5 == 4:  # Every 5 sentences, add an extra break
            md_file.write("\n")

print(f"Transcription complete! Results saved to {output_path}")
