"""Audio Clip Generator for Multilingual Assistive Navigation Alerts.

This script synthesizes spoken alert audio clips for the assistive navigation system
using Google Text-to-Speech (gTTS). Generated clips are saved as .wav files into
language-specific directories ('audio/en/' and 'audio/hi/').

Alert Dictionary Mapping:
-------------------------
- obstacle_ahead : 'Obstacle ahead'  / 'आगे रुकावट है'
- path_clear     : 'Path is clear'    / 'रास्ता साफ़ है'
- person_ahead   : 'Person ahead'    / 'आगे कोई व्यक्ति है'
- vehicle_ahead  : 'Vehicle nearby'  / 'गाड़ी पास में है'
- stairs_ahead   : 'Stairs ahead'    / 'आगे सीढ़ियाँ हैं'
- blocked_path   : 'Path is blocked' / 'रास्ता बंद है'
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_audio_clips")

# ==============================================================================
# Alert Text Mappings (English and Hindi)
# ==============================================================================
ALERT_TEXTS: Dict[str, Dict[str, str]] = {
    "obstacle_ahead": {
        "en": "Obstacle ahead",
        "hi": "आगे रुकावट है",
    },
    "path_clear": {
        "en": "Path is clear",
        "hi": "रास्ता साफ़ है",
    },
    "person_ahead": {
        "en": "Person ahead",
        "hi": "आगे कोई व्यक्ति है",
    },
    "vehicle_ahead": {
        "en": "Vehicle nearby",
        "hi": "गाड़ी पास में है",
    },
    "stairs_ahead": {
        "en": "Stairs ahead",
        "hi": "आगे सीढ़ियाँ हैं",
    },
    "blocked_path": {
        "en": "Path is blocked",
        "hi": "रास्ता बंद है",
    },
}


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """Convert an MP3 audio file to 16-bit PCM WAV format.

    Attempts conversion using:
    1. pydub (if installed and audio backend is available)
    2. ffmpeg command line binary (if present in PATH)

    Args:
        mp3_path: Path to the source .mp3 file.
        wav_path: Path to the destination .wav file.

    Returns:
        bool: True if conversion succeeded, False otherwise.
    """
    # Option 1: Try pydub
    try:
        from pydub import AudioSegment  # type: ignore

        sound = AudioSegment.from_file(str(mp3_path), format="mp3")
        sound.export(str(wav_path), format="wav")
        logger.debug("Converted via pydub: %s -> %s", mp3_path.name, wav_path.name)
        return True
    except (ImportError, Exception) as e:
        logger.debug("pydub conversion not available (%s), trying ffmpeg...", e)

    # Option 2: Try system ffmpeg
    if shutil.which("ffmpeg") is not None:
        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-loglevel",
                "error",
                "-i",
                str(mp3_path),
                "-acodec",
                "pcm_s16le",  # Standard 16-bit PCM
                "-ar",
                "44100",  # 44.1 kHz
                "-ac",
                "2",  # Stereo
                str(wav_path),
            ]
            subprocess.run(cmd, check=True)
            logger.debug("Converted via ffmpeg: %s -> %s", mp3_path.name, wav_path.name)
            return True
        except subprocess.SubprocessError as e:
            logger.error("ffmpeg conversion failed: %s", e)

    return False


def generate_clips(
    output_base_dir: Path,
    languages: tuple[str, ...] = ("en", "hi"),
    overwrite: bool = True,
) -> None:
    """Generate audio files using gTTS and save them as WAV files.

    Args:
        output_base_dir: Base directory where language folders reside (e.g. 'audio').
        languages: Tuple of language codes to generate ('en', 'hi').
        overwrite: Whether to re-synthesize clips if they already exist.
    """
    try:
        from gtts import gTTS  # type: ignore
    except ImportError:
        logger.error(
            "The 'gTTS' package is not installed. Please run:\n"
            "    pip install gTTS\n"
            "or refer to requirements.txt."
        )
        raise

    output_base_dir = Path(output_base_dir)

    for lang in languages:
        lang_dir = output_base_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating audio clips for language '%s' in '%s'...", lang, lang_dir)

        for alert_id, texts in ALERT_TEXTS.items():
            if lang not in texts:
                logger.warning("No text defined for alert '%s' in language '%s'. Skipping.", alert_id, lang)
                continue

            text_to_speak = texts[lang]
            target_wav = lang_dir / f"{alert_id}.wav"
            target_mp3 = lang_dir / f"{alert_id}.mp3"

            if target_wav.exists() and not overwrite:
                logger.info("File '%s' already exists. Skipping (overwrite=False).", target_wav.name)
                continue

            logger.info("Synthesizing [%s/%s]: '%s' -> \"%s\"", lang, alert_id, text_to_speak, target_wav.name)

            # Generate speech audio using gTTS
            tts = gTTS(text=text_to_speak, lang=lang, slow=False)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_mp3_path = Path(tmp_file.name)

            try:
                # Save MP3 stream from gTTS
                tts.save(str(tmp_mp3_path))

                # Convert MP3 to WAV
                converted = convert_mp3_to_wav(tmp_mp3_path, target_wav)

                if converted:
                    logger.info("Successfully generated: %s", target_wav)
                else:
                    # If ffmpeg and pydub are unavailable, preserve mp3
                    shutil.copyfile(tmp_mp3_path, target_mp3)
                    logger.warning(
                        "Could not convert to .wav (install ffmpeg or pydub). "
                        "Saved fallback MP3 to '%s'. Pygame mixer can play both formats.",
                        target_mp3,
                    )
            finally:
                if tmp_mp3_path.exists():
                    try:
                        os.unlink(tmp_mp3_path)
                    except OSError:
                        pass

    logger.info("Audio generation complete.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Synthesize multilingual alert audio clips (.wav) using gTTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="audio",
        help="Target base directory for audio outputs",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "hi"],
        choices=["en", "hi"],
        help="Language codes to generate clips for",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip generation if target file already exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    audio_dir = (project_root / args.output_dir).resolve()

    generate_clips(
        output_base_dir=audio_dir,
        languages=tuple(args.languages),
        overwrite=not args.no_overwrite,
    )
