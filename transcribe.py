"""
Audio transcription module using faster-whisper with loguru logging
"""

import os
import sys
import time
from pathlib import Path

import psutil
from faster_whisper import WhisperModel
from loguru import logger
from tqdm import tqdm

# Default configuration
DEFAULT_CONFIG = {
    "model_size": "base",
    "device": "cpu",
    "compute_type": "int8",
    "beam_size": 5,
    "language": "ru",
    "log_level": "INFO",
    "console_format": "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    "segment_log_interval": 10,  # Log every N segments
    "memory_log_interval": 100,  # Log memory usage every N segments
    "line_break_interval": 5,  # Add line break after N sentences
    "max_preview_chars": 50,  # Maximum characters to show in preview logs
    "show_progress": True,  # Whether to show progress bar
    "print_transcript": False,  # Whether to print transcript to console
}


# Set up console-only logger
def setup_logger(config):
    """Set up the logger with the provided configuration"""
    logger.remove()
    logger.add(
        sys.stdout,
        level=config["log_level"],
        format=config["console_format"],
        colorize=True,
        filter=lambda record: "Segment"
        not in record["message"],  # Filter out segment logs
    )
    # Add separate logger for segment details
    logger.add(
        sys.stderr,
        level="DEBUG",
        format="<dim>{time:HH:mm:ss}</dim> | <level>{level: <8}</level> | {message}",
        colorize=True,
        filter=lambda record: "Segment" in record["message"],  # Only segment logs
    )


def transcribe_audio(
    audio_path,
    output_path=None,
    model_size=None,
    language=None,
    beam_size=None,
    device=None,
    compute_type=None,
    **kwargs,
):
    """
    Transcribe audio file to text using faster-whisper

    Args:
        audio_path (str): Path to the audio file to transcribe
        output_path (str, optional): Path to save the transcription output.
            If None, will use the same directory as the audio file with a .md extension.
        model_size (str, optional): Size of the Whisper model. Defaults to config value.
        language (str, optional): Language code for transcription. Defaults to config value.
        beam_size (int, optional): Beam size for decoding. Defaults to config value.
        device (str, optional): Device to run model on. Defaults to config value.
        compute_type (str, optional): Computation type. Defaults to config value.
        **kwargs: Additional keyword arguments for configuration

    Returns:
        str: Path to the output file
    """
    # Get configuration from kwargs or use defaults
    config = DEFAULT_CONFIG.copy()
    for key in config:
        if key in kwargs:
            config[key] = kwargs[key]

    # Use defaults if not provided
    model_size = model_size or config["model_size"]
    language = language or config["language"]
    beam_size = beam_size or config["beam_size"]
    device = device or config["device"]
    compute_type = compute_type or config["compute_type"]

    # Set default output path if not provided
    if output_path is None:
        audio_file = Path(audio_path)
        output_path = str(audio_file.with_suffix(".md"))

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup logger with current config
    setup_logger(config)

    # Start timing the transcription process
    start_time = time.time()

    # Initialize the model
    logger.opt(colors=True).info(
        f"Initializing <green>Whisper</green> model: <cyan>{model_size}</cyan>"
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Log audio file info
    audio_file = Path(audio_path)
    if audio_file.exists():
        file_size = audio_file.stat().st_size / (1024 * 1024)  # Size in MB
    if audio_file.exists():
        file_size = audio_file.stat().st_size / (1024 * 1024)  # Size in MB
        logger.opt(colors=True).info(
            f"Audio file: <blue>{audio_file.name}</blue> ({file_size:.2f} MB)"
        )
    else:
        logger.opt(colors=True).error(f"Audio file <red>not found</red>: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.opt(colors=True).info(f"Output will be saved to: <blue>{output_path}</blue>")
    logger.opt(colors=True).info(
        f"Model: <cyan>{model_size}</cyan> ({compute_type}) on <yellow>{device}</yellow>"
    )

    # Create output file and write header
    logger.opt(colors=True).info("<green>Creating</green> output file...")
    with open(output_path, "w") as md_file:
        md_file.write("# Transcription Results\n\n")
        md_file.write(f"## Audio File: {audio_file.name}\n\n")
        md_file.write(f"## Model: {model_size}\n\n")
        md_file.write(f"## Language: {language}\n\n")
        md_file.write("## Transcription:\n\n")

    # Start transcription
    logger.opt(colors=True).info("<green>Starting</green> transcription process...")
    try:
        segments, info = model.transcribe(
            audio_path, beam_size=beam_size, language=language
        )
    except Exception as e:
        logger.opt(colors=True).error(f"Error during transcription: <red>{e}</red>")
        raise

    # Log audio detection info
    logger.opt(colors=True).info(
        f"Detected language: <green>{info.language}</green> (confidence: {info.language_probability:.2f})"
    )

    # Process and save each segment on the fly with progress bar
    with open(output_path, "a") as md_file:
        segment_count = 0
        total_chars = 0

        # Use a cleaner tqdm configuration
        progress_args = {
            "desc": "Transcribing",
            "unit": " segment",
            "position": 0,
            "leave": True,
            "dynamic_ncols": True,
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        }

        with (
            tqdm(**progress_args)
            if config["show_progress"]
            else tqdm(total=0, disable=True) as pbar
        ):
            for i, segment in enumerate(segments):
                segment_count += 1
                segment_text = segment.text.strip()
                segment_chars = len(segment_text)
                total_chars += segment_chars

                # Calculate transcription speed (characters per second)
                elapsed_time = time.time() - start_time
                char_rate = total_chars / elapsed_time if elapsed_time > 0 else 0

                # Log every N segments
                if i % config["segment_log_interval"] == 0:
                    preview = segment_text[: config["max_preview_chars"]]
                    if len(segment_text) > config["max_preview_chars"]:
                        preview += "..."
                    logger.opt(colors=True).info(
                        f"Segment <blue>{segment_count}</blue>: '<yellow>{preview}</yellow>' | Speed: {char_rate:.1f} chars/sec"
                    )

                # Print to console for immediate feedback (optional)
                if config["print_transcript"] and not config["show_progress"]:
                    print(f"({segment_count:03d}) {segment_text}")

                # Write to markdown file immediately
                md_file.write(f" {segment_text}\n")

                # Add a line break in markdown for readability
                if (
                    i % config["line_break_interval"]
                    == config["line_break_interval"] - 1
                ):
                    md_file.write("\n")

                # Update progress bar
                if config["show_progress"]:
                    pbar.update(1)
                    pbar.set_postfix(
                        {"speed": f"{char_rate:.1f} c/s", "seg": segment_count}
                    )

                # Memory awareness: log every N segments
                if segment_count % config["memory_log_interval"] == 0:
                    memory_usage = psutil.virtual_memory().used / (
                        1024 * 1024 * 1024
                    )  # GB
                    logger.opt(colors=True).info(
                        f"Memory: <magenta>{memory_usage:.2f} GB</magenta> | Segments: <blue>{segment_count}</blue>"
                    )

    # Calculate and log final statistics
    elapsed_time = time.time() - start_time
    logger.opt(colors=True).info("<green>✅ Transcription completed!</green>")
    logger.opt(colors=True).info(
        f"Processed <blue>{segment_count}</blue> segments with <yellow>{total_chars}</yellow> total characters"
    )
    logger.opt(colors=True).info(f"Total time: <cyan>{elapsed_time:.2f}</cyan> seconds")
    logger.opt(colors=True).info(
        f"Average speed: <cyan>{total_chars / elapsed_time:.1f}</cyan> characters per second"
    )
    logger.opt(colors=True).info(f"Results saved to: <blue>{output_path}</blue>")

    # Print final message
    if config["show_progress"]:
        print(f"\n✅ Transcription complete! Results saved to: {output_path}")

    return output_path


# Running this script directly will use command line arguments or defaults
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio file to text")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_CONFIG["model_size"],
        help=f"Model size (default: {DEFAULT_CONFIG['model_size']})",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=DEFAULT_CONFIG["language"],
        help=f"Language code (default: {DEFAULT_CONFIG['language']})",
    )
    parser.add_argument(
        "-b",
        "--beam_size",
        type=int,
        default=DEFAULT_CONFIG["beam_size"],
        help=f"Beam size (default: {DEFAULT_CONFIG['beam_size']})",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=DEFAULT_CONFIG["device"],
        help=f"Device to run model on (default: {DEFAULT_CONFIG['device']})",
    )
    parser.add_argument(
        "-c",
        "--compute_type",
        default=DEFAULT_CONFIG["compute_type"],
        help=f"Computation type (default: {DEFAULT_CONFIG['compute_type']})",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--print-transcript",
        action="store_true",
        help="Print transcript to console instead of progress bar",
    )

    args = parser.parse_args()

    # Set up logging options based on command line arguments
    kwargs = {
        "show_progress": not args.no_progress,
        "print_transcript": args.print_transcript or args.no_progress,
    }

    transcribe_audio(
        audio_path=args.audio_path,
        output_path=args.output,
        model_size=args.model,
        language=args.language,
        beam_size=args.beam_size,
        device=args.device,
        compute_type=args.compute_type,
        **kwargs,
    )
