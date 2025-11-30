"""
Audio transcription module using faster-whisper with loguru logging
"""

import os
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
    "log_rotation": "10 MB",
    "log_level": "INFO",
    "log_format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    "segment_log_interval": 10,  # Log every N segments
    "memory_log_interval": 100,  # Log memory usage every N segments
    "line_break_interval": 5,  # Add line break after N sentences
    "max_preview_chars": 50,  # Maximum characters to show in preview logs
}

# Configure logger
logger.remove()
log_file_path = os.path.join(os.path.dirname(__file__), "transcription.log")
logger.add(
    log_file_path,
    rotation=DEFAULT_CONFIG["log_rotation"],
    level=DEFAULT_CONFIG["log_level"],
    format=DEFAULT_CONFIG["log_format"],
)
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", format="{message}")


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
        **kwargs: Additional keyword arguments to pass to model.transcribe()

    Returns:
        str: Path to the output file
    """
    # Use defaults if not provided
    model_size = model_size or DEFAULT_CONFIG["model_size"]
    language = language or DEFAULT_CONFIG["language"]
    beam_size = beam_size or DEFAULT_CONFIG["beam_size"]
    device = device or DEFAULT_CONFIG["device"]
    compute_type = compute_type or DEFAULT_CONFIG["compute_type"]

    # Set default output path if not provided
    if output_path is None:
        audio_file = Path(audio_path)
        output_path = str(audio_file.with_suffix(".md"))

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start timing the transcription process
    start_time = time.time()

    # Initialize the model
    logger.info(f"Initializing Whisper model: {model_size}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Log audio file info
    audio_file = Path(audio_path)
    if audio_file.exists():
        file_size = audio_file.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"Audio file: {audio_path}")
        logger.info(f"File size: {file_size:.2f} MB")
    else:
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Output file: {output_path}")
    logger.info(f"Model: {model_size} with {compute_type} quantization on {device}")

    # Create output file and write header
    logger.info("Creating output file with headers...")
    with open(output_path, "w") as md_file:
        md_file.write("# Transcription Results\n\n")
        md_file.write(f"## Audio File: {os.path.basename(audio_path)}\n\n")
        md_file.write(f"## Model: {model_size}\n\n")
        md_file.write(f"## Language: {language}\n\n")
        md_file.write("## Transcription:\n\n")

    # Start transcription
    logger.info("Starting transcription process...")
    try:
        segments, info = model.transcribe(
            audio_path, beam_size=beam_size, language=language, **kwargs
        )
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise

    # Log audio detection info
    logger.info(
        f"Detected language: {info.language} with probability {info.language_probability:.2f}"
    )

    # Process and save each segment on the fly with progress bar
    with open(output_path, "a") as md_file:
        segment_count = 0
        total_chars = 0

        segment_log_interval = kwargs.get(
            "segment_log_interval", DEFAULT_CONFIG["segment_log_interval"]
        )
        line_break_interval = kwargs.get(
            "line_break_interval", DEFAULT_CONFIG["line_break_interval"]
        )
        max_preview_chars = kwargs.get(
            "max_preview_chars", DEFAULT_CONFIG["max_preview_chars"]
        )

        with tqdm(desc="Transcribing", unit=" segment", position=0, leave=True) as pbar:
            for i, segment in enumerate(segments):
                segment_count += 1
                segment_text = segment.text.strip()
                segment_chars = len(segment_text)
                total_chars += segment_chars

                # Calculate transcription speed (characters per second)
                elapsed_time = time.time() - start_time
                char_rate = total_chars / elapsed_time if elapsed_time > 0 else 0

                # Log every N segments to avoid spamming the log
                if i % segment_log_interval == 0:
                    preview = segment_text[:max_preview_chars]
                    if len(segment_text) > max_preview_chars:
                        preview += "..."
                    logger.info(f"Processing segment {segment_count}: '{preview}'")
                    logger.info(
                        f"Speed: {char_rate:.1f} chars/sec | Segments processed: {segment_count}"
                    )

                # Print to console for immediate feedback
                print(segment_text)

                # Write to markdown file immediately
                md_file.write(f" {segment_text}\n")

                # Add a line break in markdown for readability
                if i % line_break_interval == line_break_interval - 1:
                    md_file.write("\n")

                # Update progress bar with additional info
                pbar.update(1)
                pbar.set_description(
                    f"Segment {segment_count} | Speed: {char_rate:.1f} chars/sec"
                )

                # Memory awareness: log every N segments
                memory_log_interval = kwargs.get(
                    "memory_log_interval", DEFAULT_CONFIG["memory_log_interval"]
                )
                if segment_count % memory_log_interval == 0:
                    memory_usage = psutil.virtual_memory().used / (
                        1024 * 1024 * 1024
                    )  # GB
                    logger.info(
                        f"Memory usage: {memory_usage:.2f} GB | Segments processed: {segment_count}"
                    )

    # Calculate and log final statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Transcription complete!")
    logger.info(
        f"Processed {segment_count} segments with {total_chars} total characters"
    )
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(
        f"Average speed: {total_chars / elapsed_time:.1f} characters per second"
    )
    logger.info(f"Results saved to {output_path}")

    return output_path


def configure_logger(**kwargs):
    """
    Configure the logger with custom settings

    Args:
        **kwargs: Logger configuration options
    """
    logger.remove()

    # Update configuration with provided kwargs
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)

    log_file_path = kwargs.get(
        "log_file_path", os.path.join(os.path.dirname(__file__), "transcription.log")
    )

    logger.add(
        log_file_path,
        rotation=config.get("log_rotation", DEFAULT_CONFIG["log_rotation"]),
        level=config.get("log_level", DEFAULT_CONFIG["log_level"]),
        format=config.get("log_format", DEFAULT_CONFIG["log_format"]),
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level=config.get("log_level", DEFAULT_CONFIG["log_level"]),
        format=config.get("console_format", "{message}"),
    )


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

    args = parser.parse_args()

    # Configure logger if needed
    # configure_logger(log_level="DEBUG")  # Example

    transcribe_audio(
        audio_path=args.audio_path,
        output_path=args.output,
        model_size=args.model,
        language=args.language,
        beam_size=args.beam_size,
        device=args.device,
        compute_type=args.compute_type,
    )
