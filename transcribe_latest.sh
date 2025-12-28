#!/usr/bin/env bash

# Script to transcribe an audio file (or latest recording)
# Converts audio to WAV format first, then runs transcription

set -e  # Exit on any error

# Paths configuration
RECORDINGS_DIR="${RECORDINGS_DIR:-$HOME/recordings}"
TRANSCRIPTS_DIR="${TRANSCRIPTS_DIR:-$HOME/projects/personal/notes/private/transcripts/raw}"
TRANSCRIBE_SCRIPT="transcribe.py"  # Assumes script is in the same directory

# Optional env overrides:
#   TRANSCRIPTS_DIR=/path MODEL_SIZE=small LANGUAGE=ru ./transcribe_latest.sh [audio_file]

# Get current date and time for the output filename
current_datetime=$(date +"%Y-%m-%d_%H-%M")

# Check if recordings directory exists
if [ ! -d "$RECORDINGS_DIR" ]; then
    echo "Error: Recordings directory '$RECORDINGS_DIR' does not exist."
    exit 1
fi

# Use an explicit file if provided; otherwise pick the newest
input_file="${1:-}"

if [ -n "$input_file" ]; then
    if [ ! -f "$input_file" ]; then
        echo "Error: Input file '$input_file' does not exist."
        exit 1
    fi
    latest_file="$input_file"
else
    # Find the latest audio file in the recordings directory
    latest_file=$(ls -t "$RECORDINGS_DIR"/*.{mp3,wav,m4a,flac} 2>/dev/null | head -n 1)

    # Check if a file was found
    if [ -z "$latest_file" ]; then
        echo "No audio files found in $RECORDINGS_DIR"
        exit 1
    fi
fi

echo "Input file: $latest_file"

# Create temporary directory for processing if it doesn't exist
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT  # Clean up temporary directory on exit

# Convert to WAV format with required specifications
output_wav="$TEMP_DIR/output.wav"
echo "Converting audio to WAV format..."
ffmpeg -i "$latest_file" -ar 16000 -ac 1 -c:a pcm_s16le "$output_wav" -y -hide_banner -loglevel error

# Create transcripts directory if it doesn't exist
mkdir -p "$TRANSCRIPTS_DIR"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Prefer repo virtualenv python if available
PYTHON_BIN="python"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
fi

# Run transcription with the necessary environment variable and parameters
echo "Starting transcription..."
export OMP_NUM_THREADS=10
"$PYTHON_BIN" "$SCRIPT_DIR/$TRANSCRIBE_SCRIPT" "$output_wav" \
    -o "$TRANSCRIPTS_DIR/${current_datetime}.md" \
    -m "${MODEL_SIZE:-small}" \
    -l "${LANGUAGE:-ru}"

echo "Transcription completed successfully!"
echo "Output saved to: $TRANSCRIPTS_DIR/${current_datetime}.md"
