#!/usr/bin/env python3
"""TUI for recording and transcribing audio using Textual with animations."""

import asyncio
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    Select,
    Static,
)


class PulsingDot(Static):
    """An animated pulsing dot indicator."""

    DEFAULT_CSS = """
    PulsingDot {
        width: 3;
        height: 1;
    }
    """

    active = reactive(False)
    frame = reactive(0)

    FRAMES = ["   ", " . ", " o ", " O ", " o ", " . "]
    RECORDING_FRAMES = [
        "   ",
        " \u25cf ",
        " \u25cb ",
        " \u25cf ",
        " \u25cb ",
        " \u25cf ",
    ]

    def __init__(self, recording_mode: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.recording_mode = recording_mode
        self._anim_timer = None

    def on_mount(self) -> None:
        self._anim_timer = self.set_interval(0.15, self._tick, pause=True)

    def _tick(self) -> None:
        if self.active:
            self.frame = (self.frame + 1) % len(self.FRAMES)

    def watch_frame(self, frame: int) -> None:
        frames = self.RECORDING_FRAMES if self.recording_mode else self.FRAMES
        self.update(
            f"[bold red]{frames[frame]}[/]"
            if self.recording_mode
            else f"[bold cyan]{frames[frame]}[/]"
        )

    def start(self) -> None:
        self.active = True
        self.frame = 0
        if self._anim_timer:
            self._anim_timer.resume()

    def stop(self) -> None:
        self.active = False
        if self._anim_timer:
            self._anim_timer.pause()
        self.update("   ")


class SpinnerWidget(Static):
    """An animated spinner for processing states."""

    DEFAULT_CSS = """
    SpinnerWidget {
        width: auto;
        height: 1;
    }
    """

    active = reactive(False)
    frame = reactive(0)

    FRAMES = [
        "\u280b",
        "\u2819",
        "\u2839",
        "\u2838",
        "\u283c",
        "\u2834",
        "\u2826",
        "\u2827",
        "\u2807",
        "\u280f",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._anim_timer = None

    def on_mount(self) -> None:
        self._anim_timer = self.set_interval(0.08, self._tick, pause=True)

    def _tick(self) -> None:
        if self.active:
            self.frame = (self.frame + 1) % len(self.FRAMES)

    def watch_frame(self, frame: int) -> None:
        self.update(f"[bold cyan]{self.FRAMES[frame]}[/]")

    def start(self) -> None:
        self.active = True
        self.frame = 0
        if self._anim_timer:
            self._anim_timer.resume()

    def stop(self) -> None:
        self.active = False
        if self._anim_timer:
            self._anim_timer.pause()
        self.update(" ")


class WaveformWidget(Static):
    """Animated waveform visualization for recording."""

    DEFAULT_CSS = """
    WaveformWidget {
        width: 100%;
        height: 1;
        content-align: center middle;
    }
    """

    active = reactive(False)
    frame = reactive(0)

    BARS = [
        "\u2581",
        "\u2582",
        "\u2583",
        "\u2584",
        "\u2585",
        "\u2586",
        "\u2587",
        "\u2588",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._anim_timer = None
        self._pattern = [0] * 20
        import random

        self._random = random

    def on_mount(self) -> None:
        self._anim_timer = self.set_interval(0.1, self._tick, pause=True)

    def _tick(self) -> None:
        if self.active:
            # Shift pattern left and add new random value
            self._pattern = self._pattern[1:] + [self._random.randint(0, 7)]
            self.frame += 1
            self._update_display()

    def _update_display(self) -> None:
        bars = "".join(f"[bold red]{self.BARS[v]}[/]" for v in self._pattern)
        self.update(bars)

    def start(self) -> None:
        self.active = True
        self._pattern = [self._random.randint(0, 4) for _ in range(20)]
        if self._anim_timer:
            self._anim_timer.resume()

    def stop(self) -> None:
        self.active = False
        if self._anim_timer:
            self._anim_timer.pause()
        self.update("[dim]" + "\u2581" * 20 + "[/]")


class Timer(Static):
    """A widget that displays elapsed time with animation."""

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
        if self.running:
            # Blinking colon effect
            sep = ":" if elapsed % 2 == 0 else " "
            self.update(f"[bold]{hours:02d}{sep}{minutes:02d}{sep}{seconds:02d}[/]")
        else:
            self.update(f"[dim]{hours:02d}:{minutes:02d}:{seconds:02d}[/]")


class StatusIndicator(Static):
    """Visual status indicator with icons."""

    status = reactive("idle")

    STATUS_CONFIG = {
        "idle": ("\u25cb", "Idle", "dim white"),
        "recording": ("\u25cf", "Recording", "bold red"),
        "loading_model": ("\u2699", "Loading Model", "bold yellow"),
        "converting": ("\u21bb", "Converting Audio", "bold yellow"),
        "transcribing": ("\u270e", "Transcribing", "bold cyan"),
        "done": ("\u2714", "Complete", "bold green"),
        "error": ("\u2718", "Error", "bold red"),
        "cancelled": ("\u25a0", "Cancelled", "yellow"),
    }

    def watch_status(self, status: str) -> None:
        icon, text, style = self.STATUS_CONFIG.get(
            status, ("\u003f", "Unknown", "white")
        )
        self.update(f"[{style}]{icon} {text}[/]")


class TranscriptionProgress(Static):
    """Shows detailed transcription progress."""

    DEFAULT_CSS = """
    TranscriptionProgress {
        height: auto;
        padding: 0 1;
    }
    """

    segments = reactive(0)
    total_chars = reactive(0)
    speed = reactive(0.0)
    current_text = reactive("")

    def compose(self) -> ComposeResult:
        yield Static(id="progress-stats")
        yield Static(id="progress-preview")

    def update_progress(
        self, segments: int, total_chars: int, speed: float, preview: str
    ) -> None:
        self.segments = segments
        self.total_chars = total_chars
        self.speed = speed
        self.current_text = preview[:60] + "..." if len(preview) > 60 else preview

        try:
            stats = self.query_one("#progress-stats", Static)
            stats.update(
                f"[bold]Segments:[/] [cyan]{segments}[/] | "
                f"[bold]Characters:[/] [cyan]{total_chars:,}[/] | "
                f"[bold]Speed:[/] [green]{speed:.1f}[/] chars/sec"
            )
            preview_widget = self.query_one("#progress-preview", Static)
            preview_widget.update(f'[dim italic]"{self.current_text}"[/]')
        except NoMatches:
            pass

    def reset(self) -> None:
        self.segments = 0
        self.total_chars = 0
        self.speed = 0.0
        self.current_text = ""
        try:
            self.query_one("#progress-stats", Static).update("")
            self.query_one("#progress-preview", Static).update("")
        except NoMatches:
            pass


class PathSelector(Static):
    """A clickable path display with edit capability."""

    DEFAULT_CSS = """
    PathSelector {
        height: 3;
        padding: 0 1;
        background: $surface;
        border: solid $primary-darken-2;
    }
    
    PathSelector:hover {
        background: $surface-lighten-1;
        border: solid $primary;
    }
    
    PathSelector .path-label {
        color: $text-muted;
    }
    
    PathSelector .path-value {
        color: $text;
    }
    """

    path = reactive("", init=False)

    def __init__(self, label: str, path: str, selector_id: str, **kwargs):
        super().__init__(**kwargs)
        self.label_text = label
        self.selector_id = selector_id
        self.path = path

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]{self.label_text}[/]", classes="path-label")
        yield Static(self.path, id=f"{self.selector_id}-value", classes="path-value")

    def watch_path(self, path: str) -> None:
        try:
            value_widget = self.query_one(f"#{self.selector_id}-value", Static)
            # Truncate long paths
            display_path = path
            if len(path) > 60:
                display_path = "..." + path[-57:]
            value_widget.update(f"[cyan]{display_path}[/] [dim](click to change)[/]")
        except NoMatches:
            pass

    def on_click(self) -> None:
        self.app.push_screen(DirectoryPickerScreen(self.selector_id, Path(self.path)))


class FileSelector(Static):
    """Shows selected file with option to pick a different one."""

    DEFAULT_CSS = """
    FileSelector {
        height: 3;
        padding: 0 1;
        background: $surface;
        border: solid $primary-darken-2;
    }
    
    FileSelector:hover {
        background: $surface-lighten-1;
        border: solid $primary;
    }
    """

    file_path = reactive("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Static("[bold]Audio File[/]", classes="path-label")
        yield Static("", id="file-value", classes="path-value")

    def watch_file_path(self, path: str) -> None:
        try:
            value_widget = self.query_one("#file-value", Static)
            if path:
                name = Path(path).name
                value_widget.update(f"[green]{name}[/] [dim](click to change)[/]")
            else:
                value_widget.update(
                    "[dim]No file selected (click to browse, or record new)[/]"
                )
        except NoMatches:
            pass

    def on_click(self) -> None:
        self.app.push_screen(FilePickerScreen())


class DirectoryPickerScreen(ModalScreen):
    """A modal screen for picking directories."""

    DEFAULT_CSS = """
    DirectoryPickerScreen {
        align: center middle;
    }
    
    DirectoryPickerScreen > Container {
        width: 80%;
        height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    
    DirectoryPickerScreen DirectoryTree {
        height: 1fr;
        background: $background;
        border: solid $primary-darken-2;
    }
    
    DirectoryPickerScreen #picker-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    
    DirectoryPickerScreen Input {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, selector_id: str, initial_path: Path):
        super().__init__()
        self.selector_id = selector_id
        self.initial_path = initial_path
        self.selected_path = initial_path

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(f"[bold]Select Directory[/]")
            yield Input(value=str(self.initial_path), id="path-input")
            yield DirectoryTree(
                str(
                    self.initial_path.parent
                    if self.initial_path.exists()
                    else Path.home()
                ),
                id="dir-tree",
            )
            with Horizontal(id="picker-buttons"):
                yield Button("Select", variant="primary", id="select-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        tree = self.query_one("#dir-tree", DirectoryTree)
        tree.show_guides = True

    @on(DirectoryTree.DirectorySelected)
    def on_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self.selected_path = event.path
        self.query_one("#path-input", Input).value = str(event.path)

    @on(Input.Changed, "#path-input")
    def on_path_input_changed(self, event: Input.Changed) -> None:
        self.selected_path = Path(event.value)

    @on(Button.Pressed, "#select-btn")
    def on_select(self) -> None:
        self.dismiss({"selector_id": self.selector_id, "path": self.selected_path})

    @on(Button.Pressed, "#cancel-btn")
    def action_cancel(self) -> None:
        self.dismiss(None)


class FileItem(Static):
    """A clickable file item in the file picker."""

    class Selected(Message):
        """Message emitted when a file item is clicked."""

        def __init__(self, file_path: Path) -> None:
            super().__init__()
            self.file_path = file_path

    def __init__(self, file_path: Path, display_text: str, **kwargs):
        super().__init__(display_text, **kwargs)
        self.file_path = file_path

    def on_click(self) -> None:
        self.post_message(self.Selected(self.file_path))


class FilePickerScreen(ModalScreen):
    """A modal screen for picking audio files."""

    DEFAULT_CSS = """
    FilePickerScreen {
        align: center middle;
    }
    
    FilePickerScreen > Container {
        width: 80%;
        height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    
    FilePickerScreen #file-list {
        height: 1fr;
        background: $background;
        border: solid $primary-darken-2;
        padding: 1;
    }
    
    FilePickerScreen .file-item {
        height: 1;
        padding: 0 1;
    }
    
    FilePickerScreen .file-item:hover {
        background: $primary-darken-2;
    }
    
    FilePickerScreen .file-item.selected {
        background: $primary;
    }
    
    FilePickerScreen #picker-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self):
        super().__init__()
        self.selected_file: Path | None = None

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("[bold]Select Audio File[/]")
            yield Label("[dim]From recordings directory:[/]", id="dir-label")
            with VerticalScroll(id="file-list"):
                pass  # Files will be added on mount
            with Horizontal(id="picker-buttons"):
                yield Button("Select", variant="primary", id="select-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        self.load_files()

    def load_files(self) -> None:
        # Access main app's recordings_dir
        main_app = self.app
        recordings_dir = getattr(main_app, "recordings_dir", Path.home() / "recordings")
        self.query_one("#dir-label", Label).update(f"[dim]From:[/] {recordings_dir}")

        file_list = self.query_one("#file-list", VerticalScroll)
        file_list.remove_children()

        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"]:
            audio_files.extend(recordings_dir.glob(ext))

        audio_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if not audio_files:
            file_list.mount(Static("[dim]No audio files found[/]"))
            return

        for i, f in enumerate(audio_files[:50]):  # Limit to 50 files
            size_kb = f.stat().st_size // 1024
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            display_text = f"[bold]{f.name}[/] [dim]({size_kb} KB, {mtime})[/]"
            item = FileItem(
                f,
                display_text,
                classes="file-item",
                id=f"file-{i}",
            )
            file_list.mount(item)

    @on(FileItem.Selected)
    def on_file_selected(self, event: FileItem.Selected) -> None:
        # Deselect all
        for item in self.query(".file-item"):
            item.remove_class("selected")
        # Select clicked - find the widget that sent this message
        sender = event._sender
        if isinstance(sender, FileItem):
            sender.add_class("selected")
        self.selected_file = event.file_path

    @on(Button.Pressed, "#select-btn")
    def on_select(self) -> None:
        if self.selected_file:
            self.dismiss(self.selected_file)
        else:
            self.dismiss(None)

    @on(Button.Pressed, "#cancel-btn")
    def action_cancel(self) -> None:
        self.dismiss(None)


class RecordTranscribeTUI(App):
    """TUI for recording and transcribing audio with animations."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto auto 1fr auto auto;
    }

    #status-section {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
    }

    #status-row {
        layout: horizontal;
        height: 3;
        align: left middle;
    }

    #status-indicator {
        width: auto;
        min-width: 20;
    }

    #timer {
        width: 12;
        margin-left: 2;
    }

    #waveform {
        width: 1fr;
        margin-left: 2;
    }

    #spinner {
        width: 3;
        margin-left: 1;
    }

    #transcription-progress {
        height: auto;
        margin-top: 1;
    }

    #controls {
        height: auto;
        padding: 1 2;
        background: $surface-darken-1;
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

    #paths-container {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-top: solid $primary;
    }
    
    #paths-container PathSelector {
        margin-bottom: 1;
    }
    
    #paths-container FileSelector {
        margin-bottom: 1;
    }

    Button.recording {
        background: $error;
    }
    
    Button.recording:hover {
        background: $error-lighten-1;
    }
    
    #progress-bar-container {
        height: auto;
        padding: 0 2;
        display: none;
    }
    
    #progress-bar-container.visible {
        display: block;
    }
    
    ProgressBar {
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("r", "toggle_recording", "Record/Stop"),
        Binding("t", "transcribe", "Transcribe"),
        Binding("q", "request_quit", "Quit"),
        Binding("c", "clear_log", "Clear Log"),
        Binding("f", "pick_file", "Pick File"),
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
        self.selected_audio_file: Path | None = None
        self.transcribe_task: asyncio.Task | None = None
        self._model = None
        self._model_size = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="status-section"):
            with Horizontal(id="status-row"):
                yield StatusIndicator(id="status-indicator")
                yield Timer(id="timer")
                yield WaveformWidget(id="waveform")
                yield SpinnerWidget(id="spinner")
            yield TranscriptionProgress(id="transcription-progress")
        with Container(id="controls"):
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
                yield Label("Device:")
                yield Select(
                    [("CPU", "cpu"), ("CUDA", "cuda")],
                    value=os.environ.get("DEVICE", "cpu"),
                    id="device-select",
                )
            with Horizontal(id="buttons"):
                yield Button("Record", id="record-btn", variant="primary")
                yield Button("Transcribe", id="transcribe-btn", variant="success")
                yield Button("Pick File", id="pick-file-btn", variant="default")
        with Container(id="progress-bar-container"):
            yield ProgressBar(id="progress-bar", show_eta=False)
        with Container(id="log-container"):
            yield Log(id="log", highlight=True, auto_scroll=True)
        with Container(id="paths-container"):
            yield FileSelector(id="file-selector")
            yield PathSelector(
                "Recordings Directory",
                str(self.recordings_dir),
                "recordings",
                id="recordings-path",
            )
            yield PathSelector(
                "Transcripts Directory",
                str(self.transcripts_dir),
                "transcripts",
                id="transcripts-path",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize path displays
        self.query_one("#recordings-path", PathSelector).path = str(self.recordings_dir)
        self.query_one("#transcripts-path", PathSelector).path = str(
            self.transcripts_dir
        )

        self.log_message(
            "Ready. Press 'r' to record, 't' to transcribe, 'f' to pick a file"
        )
        self.log_message(f"Max recording duration: {self.max_seconds}s")

    def log_message(self, message: str, style: str = "") -> None:
        log = self.query_one("#log", Log)
        timestamp = datetime.now().strftime("%H:%M:%S")
        if style:
            log.write_line(f"[{style}][{timestamp}] {message}[/]")
        else:
            log.write_line(f"[dim][{timestamp}][/] {message}")

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

    def action_transcribe(self) -> None:
        if self.is_recording:
            self.log_message("Cannot transcribe while recording", "yellow")
            return
        if self.is_transcribing:
            self.log_message("Transcription already in progress", "yellow")
            return

        # Use selected file, current recording, or find latest
        audio_file = self.selected_audio_file or self.current_recording

        if not audio_file:
            # Find latest audio file
            audio_files = []
            for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"]:
                audio_files.extend(self.recordings_dir.glob(ext))

            if not audio_files:
                self.log_message(
                    f"No audio files found in {self.recordings_dir}", "red"
                )
                return

            audio_file = max(audio_files, key=lambda p: p.stat().st_mtime)

        self.start_transcription(audio_file)

    def action_pick_file(self) -> None:
        self.push_screen(FilePickerScreen(), self.on_file_picked)

    def on_file_picked(self, result: Path | None) -> None:
        if result:
            self.selected_audio_file = result
            self.query_one("#file-selector", FileSelector).file_path = str(result)
            self.log_message(f"Selected file: {result.name}", "green")

    def start_recording(self) -> None:
        if self.is_recording:
            return

        if self.is_transcribing:
            self.log_message(
                "Cannot record while transcription is in progress", "yellow"
            )
            return

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.current_recording = self.recordings_dir / f"{stamp}.wav"
        self.selected_audio_file = None  # Clear any previous selection

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
            self.log_message("Error: 'rec' (sox) not found. Please install sox.", "red")
            self.query_one("#status-indicator", StatusIndicator).status = "error"
            return

        self.log_message(f"Recording to: {self.current_recording.name}", "bold red")
        self.query_one("#status-indicator", StatusIndicator).status = "recording"
        self.query_one("#timer", Timer).start()
        self.query_one("#waveform", WaveformWidget).start()

        record_btn = self.query_one("#record-btn", Button)
        record_btn.label = "Stop"
        record_btn.add_class("recording")

        # Update file selector
        self.query_one("#file-selector", FileSelector).file_path = str(
            self.current_recording
        )

        # Monitor recording process
        self.set_timer(1, self.check_recording_status)

    def check_recording_status(self) -> None:
        if self.recording_process is None:
            return

        ret = self.recording_process.poll()
        if ret is not None:
            self.finish_recording()
        else:
            self.set_timer(1, self.check_recording_status)

    def stop_recording(self) -> None:
        if self.recording_process is None:
            return

        try:
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
        self.query_one("#waveform", WaveformWidget).stop()

        record_btn = self.query_one("#record-btn", Button)
        record_btn.label = "Record"
        record_btn.remove_class("recording")

        if self.current_recording and self.current_recording.exists():
            size = self.current_recording.stat().st_size
            self.log_message(
                f"Recording saved: {self.current_recording.name} ({size // 1024} KB)",
                "green",
            )
            self.query_one("#status-indicator", StatusIndicator).status = "idle"
            self.query_one("#file-selector", FileSelector).file_path = str(
                self.current_recording
            )
            # Auto-start transcription
            self.start_transcription(self.current_recording)
        else:
            self.log_message("Recording failed or was cancelled", "red")
            self.query_one("#status-indicator", StatusIndicator).status = "error"

    def start_transcription(self, audio_file: Path) -> None:
        self.log_message(f"Starting transcription: {audio_file.name}", "cyan")
        self.query_one("#spinner", SpinnerWidget).start()

        model_size = str(self.query_one("#model-select", Select).value)
        language = str(self.query_one("#language-select", Select).value)
        device = str(self.query_one("#device-select", Select).value)

        self.transcribe_task = asyncio.create_task(
            self.run_transcription(audio_file, model_size, language, device)
        )

    async def run_transcription(
        self, audio_file: Path, model_size: str, language: str, device: str
    ) -> None:
        """Run transcription using the transcribe module directly."""
        from faster_whisper import WhisperModel

        start_time = time.time()
        status_indicator = self.query_one("#status-indicator", StatusIndicator)
        progress_widget = self.query_one(
            "#transcription-progress", TranscriptionProgress
        )
        spinner = self.query_one("#spinner", SpinnerWidget)

        try:
            # Convert audio if needed
            status_indicator.status = "converting"
            self.log_message("Converting audio to compatible format...")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_wav = Path(temp_dir) / "converted.wav"

                # Run ffmpeg conversion in a thread
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-i",
                    str(audio_file),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(temp_wav),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.wait()

                if process.returncode != 0:
                    self.log_message("Audio conversion failed", "red")
                    status_indicator.status = "error"
                    spinner.stop()
                    return

                self.log_message("Audio converted successfully", "green")

                # Load model
                status_indicator.status = "loading_model"
                self.log_message(f"Loading Whisper model: {model_size}...")

                # Check if we need to reload the model
                if self._model is None or self._model_size != model_size:
                    # Run model loading in a thread to not block UI
                    self._model = await asyncio.to_thread(
                        WhisperModel, model_size, device=device, compute_type="int8"
                    )
                    self._model_size = model_size

                self.log_message(f"Model loaded ({device})", "green")

                # Start transcription
                status_indicator.status = "transcribing"
                self.log_message("Transcribing audio...")

                # Run transcription in a thread
                lang_param = None if language == "auto" else language
                segments_gen, info = await asyncio.to_thread(
                    self._model.transcribe,
                    str(temp_wav),
                    beam_size=5,
                    language=lang_param,
                )

                self.log_message(
                    f"Detected language: {info.language} (confidence: {info.language_probability:.2f})",
                    "cyan",
                )

                # Process segments
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                output_path = self.transcripts_dir / f"{timestamp}.md"

                segment_count = 0
                total_chars = 0
                all_text = []

                # Open output file
                with open(output_path, "w") as md_file:
                    md_file.write("# Transcription Results\n\n")
                    md_file.write(f"**Audio File:** {audio_file.name}\n\n")
                    md_file.write(f"**Model:** {model_size}\n\n")
                    md_file.write(f"**Language:** {info.language}\n\n")
                    md_file.write("---\n\n")

                    # Process segments in batches to update UI
                    segments_list = await asyncio.to_thread(list, segments_gen)

                    for segment in segments_list:
                        segment_count += 1
                        text = segment.text.strip()
                        total_chars += len(text)
                        all_text.append(text)

                        elapsed = time.time() - start_time
                        speed = total_chars / elapsed if elapsed > 0 else 0

                        # Update progress
                        progress_widget.update_progress(
                            segment_count, total_chars, speed, text
                        )

                        # Write to file
                        md_file.write(f"{text}\n")
                        if segment_count % 5 == 0:
                            md_file.write("\n")

                        # Allow UI to update
                        await asyncio.sleep(0)

                # Final stats
                elapsed = time.time() - start_time
                self.log_message(
                    f"Transcription complete: {segment_count} segments, "
                    f"{total_chars:,} chars in {elapsed:.1f}s",
                    "bold green",
                )
                self.log_message(f"Output saved to: {output_path.name}", "green")

                status_indicator.status = "done"
                spinner.stop()

        except asyncio.CancelledError:
            self.log_message("Transcription cancelled", "yellow")
            status_indicator.status = "cancelled"
            spinner.stop()
            progress_widget.reset()
        except Exception as e:
            self.log_message(f"Transcription error: {e}", "red")
            status_indicator.status = "error"
            spinner.stop()
            progress_widget.reset()

    def on_directory_picker_screen_dismiss(self, result) -> None:
        if result:
            selector_id = result["selector_id"]
            path = result["path"]

            if selector_id == "recordings":
                self.recordings_dir = path
                self.query_one("#recordings-path", PathSelector).path = str(path)
                self.log_message(f"Recordings directory: {path}", "green")
            elif selector_id == "transcripts":
                self.transcripts_dir = path
                self.query_one("#transcripts-path", PathSelector).path = str(path)
                self.log_message(f"Transcripts directory: {path}", "green")

    @on(Button.Pressed, "#record-btn")
    def handle_record_button(self) -> None:
        self.action_toggle_recording()

    @on(Button.Pressed, "#transcribe-btn")
    def handle_transcribe_button(self) -> None:
        self.action_transcribe()

    @on(Button.Pressed, "#pick-file-btn")
    def handle_pick_file_button(self) -> None:
        self.action_pick_file()

    def action_clear_log(self) -> None:
        self.query_one("#log", Log).clear()

    def action_request_quit(self) -> None:
        if self.is_recording:
            self.stop_recording()
        if self.transcribe_task and not self.transcribe_task.done():
            self.transcribe_task.cancel()
        self.exit()


def main():
    app = RecordTranscribeTUI()
    app.run()


if __name__ == "__main__":
    main()
