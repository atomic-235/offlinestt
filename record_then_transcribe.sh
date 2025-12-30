#!/usr/bin/env bash

# Record from default audio input and transcribe.
# Stops recording after max duration (default 50 min) or Ctrl+C.
#
# Dependencies:
#   - sox (rec)
#   - ffmpeg (used by transcribe_latest.sh)
#
# Optional environment overrides:
#   RECORDINGS_DIR=/path
#   TRANSCRIPTS_DIR=/path (default: ~/projects/personal/notes/private/transcripts/raw)
#   MODEL_SIZE=small
#   LANGUAGE=ru
#   MAX_SECONDS=3000
#
# Usage:
#   ./record_then_transcribe.sh

set -euo pipefail

RECORDINGS_DIR="${RECORDINGS_DIR:-$HOME/recordings}"
MAX_SECONDS="${MAX_SECONDS:-3000}"          # 50min

mkdir -p "$RECORDINGS_DIR"

stamp=$(date +"%Y-%m-%d_%H-%M")
out_wav="$RECORDINGS_DIR/${stamp}.wav"

echo "Recording to: $out_wav"
echo "Max duration: ${MAX_SECONDS}s. Press Ctrl+C to stop."

# Record 16kHz mono 16-bit PCM WAV.
# Use --foreground so Ctrl+C reaches rec properly.
timeout --foreground "$MAX_SECONDS" rec -r 16000 -c 1 -b 16 "$out_wav" || true

echo "Recording finished: $out_wav"

# Hand off to existing transcription pipeline.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
"$SCRIPT_DIR/transcribe_latest.sh" "$out_wav"
