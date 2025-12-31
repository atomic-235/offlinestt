#!/usr/bin/env python3
"""TUI for recording and transcribing audio using Textual."""

import asyncio
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, Log, Select, Static


class Timer(Static):
    """A widget that displays elapsed time."""

    elapsed = reactive(0)
    running = reactive(False)

    def on_mount(self) -> None:
        self.update_timer = self.set_interval(1, self.tick, pause=True)

    def tick(self) -> None:
        if self.running:
            self.elapsed += 1

    def start(self) -> None:
        self.elapsed = 0
        self.running = True
        self.update_timer.resume()

    def stop(self) -> None:
        self.running = False
        self.update_timer.pause()

    def reset(self) -> None:
        self.elapsed = 0
        self.running = False
        self.update_timer.pause()

    def watch_elapsed(self, elapsed: int) -> None:
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        self.update(f"{hours:02d}:{minutes:02d}:{seconds:02d}")


class StatusIndicator(Static):
    """Visual status indicator."""

    status = reactive("idle")

    STATUS_STYLES = {
        "idle": ("[ IDLE ]", "dim"),
        "recording": ("[ REC ]", "bold red"),
        "transcribing": ("[ PROCESSING ]", "bold yellow"),
        "done": ("[ DONE ]", "bold green"),
        "error": ("[ ERROR ]", "bold red reverse"),
    }

    def watch_status(self, status: str) -> None:
        text, style = self.STATUS_STYLES.get(status, ("[ ? ]", ""))
        self.update(f"[{style}]{text}[/]")


class RecordTranscribeTUI(App):
    """TUI for recording and transcribing audio."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto 1fr auto;
    }

    #controls {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
    }

    #status-bar {
        layout: horizontal;
        height: 3;
        align: center middle;
        margin-bottom: 1;
    }

    #status-indicator {
        width: auto;
        margin-right: 2;
    }

    #timer {
        width: auto;
        text-style: bold;
    }

    #settings {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }

    #settings Label {
        width: auto;
        margin-right: 1;
        padding-top: 1;
    }

    #settings Select {
        width: 20;
        margin-right: 2;
    }

    #buttons {
        layout: horizontal;
        height: auto;
        align: center middle;
    }

    #buttons Button {
        margin: 0 1;
    }

    #log-container {
        padding: 1 2;
    }

    Log {
        background: $surface;
        border: solid $primary;
    }

    #info-bar {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-top: solid $primary;
    }

    #paths-info {
        height: auto;
    }

    .path-label {
        color: $text-muted;
    }

    .path-value {
        color: $text;
    }

    Button.recording {
        background: $error;
    }
    """

    BINDINGS = [
        ("r", "toggle_recording", "Record/Stop"),
        ("t", "transcribe_latest", "Transcribe Latest"),
        ("q", "quit", "Quit"),
        ("c", "clear_log", "Clear Log"),
    ]

    TITLE = "Offline STT"
    SUB_TITLE = "Record & Transcribe"

    def __init__(self):
        super().__init__()
        self.recordings_dir = Path(
            os.environ.get("RECORDINGS_DIR", Path.home() / "recordings")
        )
        self.transcripts_dir = Path(
            os.environ.get(
                "TRANSCRIPTS_DIR",
                Path.home() / "projects/personal/notes/private/transcripts/raw",
            )
        )
        self.max_seconds = int(os.environ.get("MAX_SECONDS", 3000))
        self.recording_process: subprocess.Popen | None = None
        self.current_recording: Path | None = None
        self.transcribe_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="controls"):
            with Horizontal(id="status-bar"):
                yield StatusIndicator(id="status-indicator")
                yield Timer(id="timer")
            with Horizontal(id="settings"):
                yield Label("Model:")
                yield Select(
                    [(s, s) for s in ["tiny", "small", "medium", "large-v3"]],
                    value=os.environ.get("MODEL_SIZE", "medium"),
                    id="model-select",
                )
                yield Label("Language:")
                yield Select(
                    [("Russian", "ru"), ("English", "en"), ("Auto", "auto")],
                    value=os.environ.get("LANGUAGE", "ru"),
                    id="language-select",
                )
            with Horizontal(id="buttons"):
                yield Button("Record", id="record-btn", variant="primary")
                yield Button(
                    "Transcribe Latest", id="transcribe-btn", variant="default"
                )
        with Container(id="log-container"):
            yield Log(id="log", highlight=True)
        with Container(id="info-bar"):
            yield Static(id="paths-info")
        yield Footer()

    def on_mount(self) -> None:
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.update_paths_info()
        self.log_message(f"Ready. Max recording duration: {self.max_seconds}s")
        self.log_message("Press 'r' to start recording, 't' to transcribe latest file")

    def update_paths_info(self) -> None:
        paths_info = self.query_one("#paths-info", Static)
        paths_info.update(
            f"[dim]Recordings:[/] {self.recordings_dir}\n"
            f"[dim]Transcripts:[/] {self.transcripts_dir}"
        )

    def log_message(self, message: str) -> None:
        log = self.query_one("#log", Log)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.write_line(f"[{timestamp}] {message}")

    @property
    def is_recording(self) -> bool:
        return self.recording_process is not None

    @property
    def is_transcribing(self) -> bool:
        return self.transcribe_task is not None and not self.transcribe_task.done()

    def action_toggle_recording(self) -> None:
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self.is_recording:
            return

        if self.is_transcribing:
            self.log_message("Cannot record while transcription is in progress")
            return

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.current_recording = self.recordings_dir / f"{stamp}.wav"

        try:
            self.recording_process = subprocess.Popen(
                [
                    "timeout",
                    "--foreground",
                    str(self.max_seconds),
                    "rec",
                    "-r",
                    "16000",
                    "-c",
                    "1",
                    "-b",
                    "16",
                    str(self.current_recording),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            self.log_message("Error: 'rec' (sox) not found. Please install sox.")
            self.query_one("#status-indicator", StatusIndicator).status = "error"
            return

        self.log_message(f"Recording to: {self.current_recording}")
        self.query_one("#status-indicator", StatusIndicator).status = "recording"
        self.query_one("#timer", Timer).start()

        record_btn = self.query_one("#record-btn", Button)
        record_btn.label = "Stop"
        record_btn.add_class("recording")

        # Monitor recording process
        self.set_timer(1, self.check_recording_status)

    def check_recording_status(self) -> None:
        if self.recording_process is None:
            return

        ret = self.recording_process.poll()
        if ret is not None:
            # Recording finished (timeout or process ended)
            self.finish_recording()
        else:
            self.set_timer(1, self.check_recording_status)

    def stop_recording(self) -> None:
        if self.recording_process is None:
            return

        try:
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(self.recording_process.pid), signal.SIGTERM)
            self.recording_process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.recording_process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

        self.finish_recording()

    def finish_recording(self) -> None:
        self.recording_process = None
        self.query_one("#timer", Timer).stop()

        record_btn = self.query_one("#record-btn", Button)
        record_btn.label = "Record"
        record_btn.remove_class("recording")

        if self.current_recording and self.current_recording.exists():
            size = self.current_recording.stat().st_size
            self.log_message(
                f"Recording saved: {self.current_recording.name} ({size // 1024} KB)"
            )
            self.query_one("#status-indicator", StatusIndicator).status = "idle"
            # Auto-start transcription
            self.start_transcription(self.current_recording)
        else:
            self.log_message("Recording failed or was cancelled")
            self.query_one("#status-indicator", StatusIndicator).status = "error"

    def action_transcribe_latest(self) -> None:
        if self.is_recording:
            self.log_message("Cannot transcribe while recording")
            return

        if self.is_transcribing:
            self.log_message("Transcription already in progress")
            return

        # Find latest audio file
        audio_files = list(self.recordings_dir.glob("*.wav"))
        audio_files.extend(self.recordings_dir.glob("*.mp3"))
        audio_files.extend(self.recordings_dir.glob("*.m4a"))
        audio_files.extend(self.recordings_dir.glob("*.flac"))

        if not audio_files:
            self.log_message(f"No audio files found in {self.recordings_dir}")
            return

        latest = max(audio_files, key=lambda p: p.stat().st_mtime)
        self.start_transcription(latest)

    def start_transcription(self, audio_file: Path) -> None:
        self.log_message(f"Starting transcription: {audio_file.name}")
        self.query_one("#status-indicator", StatusIndicator).status = "transcribing"

        model = self.query_one("#model-select", Select).value
        language = self.query_one("#language-select", Select).value

        self.transcribe_task = asyncio.create_task(
            self.run_transcription(audio_file, str(model), str(language))
        )

    async def run_transcription(
        self, audio_file: Path, model: str, language: str
    ) -> None:
        script_dir = Path(__file__).parent
        transcribe_script = script_dir / "transcribe_latest.sh"

        env = os.environ.copy()
        env["MODEL_SIZE"] = model
        env["LANGUAGE"] = language
        env["RECORDINGS_DIR"] = str(self.recordings_dir)
        env["TRANSCRIPTS_DIR"] = str(self.transcripts_dir)

        try:
            process = await asyncio.create_subprocess_exec(
                str(transcribe_script),
                str(audio_file),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            if process.stdout:
                async for line in process.stdout:
                    text = line.decode().strip()
                    if text:
                        self.log_message(text)

            await process.wait()

            if process.returncode == 0:
                self.log_message("Transcription completed successfully")
                self.query_one("#status-indicator", StatusIndicator).status = "done"
            else:
                self.log_message(f"Transcription failed with code {process.returncode}")
                self.query_one("#status-indicator", StatusIndicator).status = "error"

        except Exception as e:
            self.log_message(f"Transcription error: {e}")
            self.query_one("#status-indicator", StatusIndicator).status = "error"

    @on(Button.Pressed, "#record-btn")
    def handle_record_button(self) -> None:
        self.action_toggle_recording()

    @on(Button.Pressed, "#transcribe-btn")
    def handle_transcribe_button(self) -> None:
        self.action_transcribe_latest()

    def action_clear_log(self) -> None:
        self.query_one("#log", Log).clear()

    def action_quit(self) -> None:
        if self.is_recording:
            self.stop_recording()
        self.exit()


def main():
    app = RecordTranscribeTUI()
    app.run()


if __name__ == "__main__":
    main()
