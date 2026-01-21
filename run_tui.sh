#!/usr/bin/env bash
# Wrapper script to run the TUI with environment variables

set -euo pipefail

cd /home/antonym/projects/utils/offlinestt

exec bash -i -c 'uv run python tui.py'
