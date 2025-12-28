#!/usr/bin/env bash

# Record from default audio input and transcribe.
# Stops recording after 50 minutes OR after 5 minutes of silence.
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
#   SILENCE_DB=-45dB
#   MAX_SECONDS=3000
#   SILENCE_SECONDS=300
#
# Usage:
#   ./record_then_transcribe.sh

set -euo pipefail

RECORDINGS_DIR="${RECORDINGS_DIR:-$HOME/recordings}"
MAX_SECONDS="${MAX_SECONDS:-3000}"          # 50min
SILENCE_SECONDS="${SILENCE_SECONDS:-300}"   # 5min
SILENCE_DB="${SILENCE_DB:--45dB}"

mkdir -p "$RECORDINGS_DIR"

stamp=$(date +"%Y-%m-%d_%H-%M")
out_wav="$RECORDINGS_DIR/${stamp}.wav"

echo "Recording to: $out_wav"
echo "Stop conditions: ${MAX_SECONDS}s max OR ${SILENCE_SECONDS}s silence < ${SILENCE_DB}"
echo "(Adjust SILENCE_DB if it stops too early/late.)"

# Record 16kHz mono 16-bit PCM WAV.
#
# silence effect form (SoX):
#   silence [above-periods] [above-duration] [above-threshold] [below-periods] [below-duration] [below-threshold]
# Here we stop after SILENCE_SECONDS below SILENCE_DB.
rec -q -r 16000 -c 1 -b 16 "$out_wav" \
  silence 1 0.1 "$SILENCE_DB" 1 "$SILENCE_SECONDS" "$SILENCE_DB" \
  trim 0 "$MAX_SECONDS"

echo "Recording finished: $out_wav"

# Hand off to existing transcription pipeline.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
"$SCRIPT_DIR/transcribe_latest.sh" "$out_wav"
