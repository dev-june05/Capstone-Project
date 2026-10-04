"""Audio Alert System for Assistive Navigation on Jetson Orin Nano.

This module manages non-blocking, queued audio notifications played through
a dedicated background worker thread using pygame.mixer. It prevents alert spamming
by enforcing configurable cooldown timers per alert type.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import pygame

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Supported Alert Definitions
# ==============================================================================
ALERTS: Dict[str, str] = {
    "obstacle_ahead": "General obstacle detected directly in the navigation path",
    "path_clear": "Navigation pathway is clear of detected obstacles",
    "person_ahead": "Pedestrian or person detected within close proximity",
    "vehicle_ahead": "Vehicle detected nearby or moving toward walking path",
    "stairs_ahead": "Stairs, curb, or elevation hazard detected ahead",
    "blocked_path": "Walking path is completely blocked or impassable",
}


class AudioAlertSystem:
    """Threaded audio alert system with cooldown enforcement for edge devices.

    Manages pre-loaded audio notification clips (.wav format) and plays them
    asynchronously using a background worker thread. Guarantees that live
    vision inference loops are never blocked by audio I/O.

    Attributes:
        audio_dir (Path): Base directory containing language-specific audio subdirectories.
        lang (str): Active language code ('en' for English, 'hi' for Hindi).
        cooldown (float): Minimum duration (in seconds) before the same alert can replay.
        sounds (Dict[str, pygame.mixer.Sound]): Cache of loaded sound objects keyed by alert ID.
        alert_queue (queue.Queue): Thread-safe queue for incoming playback requests.
        last_played_times (Dict[str, float]): Timestamp map of last played time per alert ID.
        is_running (threading.Event): Thread execution control flag.
        worker_thread (Optional[threading.Thread]): Background playback worker thread.
    """

    def __init__(
        self,
        audio_dir: str | Path = "audio",
        lang: str = "en",
        cooldown: float = 4.0,
    ) -> None:
        """Initialize the AudioAlertSystem, audio mixer, and sound cache.

        Args:
            audio_dir: Path to directory containing language subfolders (e.g. 'audio/en').
            lang: Language code ('en' or 'hi'). Default is 'en'.
            cooldown: Cooldown interval in seconds between repeated alerts. Default is 4.0.
        """
        self.audio_dir = Path(audio_dir)
        self.lang = lang.lower()
        self.cooldown = float(cooldown)

        if self.lang not in ("en", "hi"):
            logger.warning(
                "Language '%s' is not officially supported ('en'/'hi'). Falling back to 'en'.",
                self.lang,
            )
            self.lang = "en"

        self.alert_queue: queue.Queue[str] = queue.Queue(maxsize=16)
        self.last_played_times: Dict[str, float] = {}
        self.sounds: Dict[str, pygame.mixer.Sound] = {}

        self.is_running = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        self._init_mixer()
        self._load_sounds()

    def _init_mixer(self) -> None:
        """Initialize the Pygame audio mixer safely."""
        try:
            if not pygame.mixer.get_init():
                # Initialize pygame mixer with 44.1kHz, 16-bit stereo, 2 channels
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
                logger.info("Pygame mixer initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize pygame.mixer: %s", e)
            logger.info("Ensure ALSA/PulseAudio is running or export SDL_AUDIODRIVER=dummy.")

    def _load_sounds(self) -> None:
        """Load all .wav alert audio files for the specified language into memory."""
        lang_dir = self.audio_dir / self.lang

        if not lang_dir.exists():
            logger.warning("Audio directory '%s' does not exist. Creating directory.", lang_dir)
            lang_dir.mkdir(parents=True, exist_ok=True)

        for alert_id in ALERTS:
            wav_path = lang_dir / f"{alert_id}.wav"
            mp3_path = lang_dir / f"{alert_id}.mp3"

            sound_file = None
            if wav_path.is_file():
                sound_file = wav_path
            elif mp3_path.is_file():
                sound_file = mp3_path

            if sound_file:
                try:
                    self.sounds[alert_id] = pygame.mixer.Sound(str(sound_file))
                    logger.debug("Loaded audio clip: %s -> %s", alert_id, sound_file)
                except Exception as e:
                    logger.error("Error loading sound file %s: %s", sound_file, e)
            else:
                logger.warning(
                    "Missing sound file for alert '%s' at '%s'. Alerts will be logged to console.",
                    alert_id,
                    wav_path,
                )

        logger.info(
            "Loaded %d/%d alert audio clips for language '%s'.",
            len(self.sounds),
            len(ALERTS),
            self.lang,
        )

    def play(self, alert_id: str) -> bool:
        """Enqueue an alert to be played if its cooldown period has elapsed.

        This method is strictly non-blocking. It returns immediately to ensure
        the visual inference pipeline is not interrupted.

        Args:
            alert_id: Identifier of the alert to play (must be a key in ALERTS).

        Returns:
            bool: True if the alert was successfully enqueued, False if suppressed
                  by cooldown, unknown alert ID, or full queue.
        """
        if alert_id not in ALERTS:
            logger.warning("Attempted to play unknown alert ID: '%s'", alert_id)
            return False

        current_time = time.time()
        last_played = self.last_played_times.get(alert_id, 0.0)

        # Check if the cooldown interval has passed
        if current_time - last_played < self.cooldown:
            # Cooldown active, suppress alert
            return False

        try:
            self.alert_queue.put_nowait(alert_id)
            self.last_played_times[alert_id] = current_time
            logger.info("Queued alert: '%s' (%s)", alert_id, ALERTS[alert_id])
            return True
        except queue.Full:
            logger.warning("Alert queue is full. Dropping alert: '%s'", alert_id)
            return False

    def _run(self) -> None:
        """Internal worker loop executed in a background daemon thread.

        Continuously monitors the alert queue, dequeues alert IDs, and plays
        audio using pygame.mixer. Blocks waiting on queue with a short timeout
        to allow clean thread termination.
        """
        logger.info("AudioAlertSystem background worker started.")

        while self.is_running.is_set():
            try:
                # Wait up to 0.2s for an alert ID so we can react to stop signals
                alert_id = self.alert_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Play the sound clip
            sound = self.sounds.get(alert_id)
            if sound is not None:
                try:
                    channel = sound.play()
                    # Wait for playback to finish while keeping responsiveness to stop event
                    if channel is not None:
                        while channel.get_busy() and self.is_running.is_set():
                            time.sleep(0.05)
                except Exception as e:
                    logger.error("Error during playback of alert '%s': %s", alert_id, e)
            else:
                # Fallback if no audio clip file was present
                logger.info("[AUDIO FALLBACK] BEEP: Alert '%s' triggered (%s)", alert_id, ALERTS.get(alert_id))
                time.sleep(0.5)

            self.alert_queue.task_done()

        logger.info("AudioAlertSystem background worker exiting.")

    def start(self) -> None:
        """Start the background audio playback thread."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            logger.warning("AudioAlertSystem thread is already running.")
            return

        self.is_running.set()
        self.worker_thread = threading.Thread(
            target=self._run,
            name="AudioAlertWorkerThread",
            daemon=True,
        )
        self.worker_thread.start()
        logger.info("AudioAlertSystem started.")

    def stop(self) -> None:
        """Stop the background audio playback thread and release resources."""
        if not self.is_running.is_set():
            return

        logger.info("Stopping AudioAlertSystem...")
        self.is_running.clear()

        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None

        try:
            if pygame.mixer.get_init():
                pygame.mixer.stop()
                pygame.mixer.quit()
        except Exception as e:
            logger.error("Error shutting down pygame.mixer: %s", e)

        logger.info("AudioAlertSystem stopped cleanly.")


# ==============================================================================
# Quick Standalone Test
# ==============================================================================
if __name__ == "__main__":
    print("Testing AudioAlertSystem...")
    alert_sys = AudioAlertSystem(audio_dir="audio", lang="en", cooldown=2.0)
    alert_sys.start()

    try:
        print("Triggering alert: obstacle_ahead")
        alert_sys.play("obstacle_ahead")
        time.sleep(1.0)

        print("Triggering alert again (should be suppressed by cooldown): obstacle_ahead")
        alert_sys.play("obstacle_ahead")
        time.sleep(2.0)

        print("Triggering alert after cooldown: obstacle_ahead")
        alert_sys.play("obstacle_ahead")
        time.sleep(2.0)

        print("Triggering alert: path_clear")
        alert_sys.play("path_clear")
        time.sleep(1.5)
    finally:
        alert_sys.stop()
        print("Test complete.")
