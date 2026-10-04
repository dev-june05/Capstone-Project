"""Live Semantic Segmentation Inference on NVIDIA Jetson Orin Nano.

This script opens a camera feed via GStreamer (v4l2src for USB webcam),
processes frames through an optimized TensorRT engine (BiSeNetV2 / Fast-SCNN),
analyzes the central walking zone for 4-class semantic hazards
(Walkable, Non-walkable, Dynamic Obstacle, Structural Obstacle),
and triggers non-blocking multilingual audio alerts via AudioAlertSystem.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Import the audio alert system
try:
    from jetson.audio_alerts import AudioAlertSystem
except ImportError:
    # Allow execution directly from within the jetson directory
    from audio_alerts import AudioAlertSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("infer_live")

# ==============================================================================
# Class Definitions and Color Palette (4 Classes)
# ==============================================================================
# 0: Walkable            -> Green  [0, 255, 0]
# 1: Non-walkable        -> Gray   [128, 128, 128]
# 2: Dynamic Obstacle    -> Red    [0, 0, 255]
# 3: Structural Obstacle -> Yellow [0, 255, 255]
CLASS_NAMES = ["Walkable", "Non-walkable", "Dynamic Obstacle", "Structural Obstacle"]

COLOR_MAP = np.array(
    [
        [0, 255, 0],    # Class 0: Walkable (Green)
        [128, 128, 128],# Class 1: Non-walkable (Gray)
        [0, 0, 255],    # Class 2: Dynamic Obstacle (Red)
        [0, 255, 255],  # Class 3: Structural Obstacle (Yellow)
    ],
    dtype=np.uint8,
)


# ==============================================================================
# GStreamer Pipeline Builders for Jetson
# ==============================================================================
def build_gstreamer_pipeline_usb(
    device_id: int = 0,
    width: int = 1280,
    height: int = 720,
    framerate: int = 30,
) -> str:
    """Build a GStreamer pipeline string for standard USB V4L2 webcams.

    Args:
        device_id: V4L2 camera device index (/dev/video<id>).
        width: Capture frame width.
        height: Capture frame height.
        framerate: Video stream frames per second.

    Returns:
        str: GStreamer pipeline string for cv2.VideoCapture.
    """
    return (
        f"v4l2src device=/dev/video{device_id} ! "
        f"video/x-raw, width={width}, height={height}, framerate={framerate}/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )


def build_gstreamer_pipeline_csi(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 1280,
    display_height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """Build a GStreamer pipeline string for Jetson CSI MIPI cameras via nvarguscamerasrc.

    Args:
        sensor_id: CSI camera hardware sensor ID (default: 0).
        capture_width: Sensor capture resolution width.
        capture_height: Sensor capture resolution height.
        display_width: Output frame width.
        display_height: Output frame height.
        framerate: Sensor framerate.
        flip_method: Hardware image rotation/flip (0: none, 2: 180 deg).

    Returns:
        str: GStreamer pipeline string for cv2.VideoCapture.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )


# ==============================================================================
# TensorRT Engine Execution Wrapper
# ==============================================================================
class TensorRTInference:
    """Wrapper for deserializing and executing TensorRT inference engines on Jetson."""

    def __init__(self, engine_path: str | Path, input_shape: Tuple[int, int] = (512, 512)) -> None:
        """Initialize TensorRT runtime, deserialize engine, and allocate CUDA memory.

        Args:
            engine_path: Path to serialized .engine plan file.
            input_shape: Expected network input spatial resolution (Height, Width).
        """
        self.engine_path = Path(engine_path)
        self.input_shape = input_shape  # (H, W)
        self.context = None
        self.engine = None
        self.stream = None
        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []

        logger.info("Initializing TensorRT engine from: %s", self.engine_path)
        self._load_engine()
        self._allocate_buffers()

    def _load_engine(self) -> None:
        """Deserialize TensorRT runtime engine from file.

        TODO: Implement TensorRT runtime engine deserialization:
            1. import tensorrt as trt
            2. trt_logger = trt.Logger(trt.Logger.WARNING)
            3. runtime = trt.Runtime(trt_logger)
            4. with open(self.engine_path, 'rb') as f:
                   self.engine = runtime.deserialize_cuda_engine(f.read())
            5. self.context = self.engine.create_execution_context()
        """
        # TODO: Replace with real TensorRT deserialization on Jetson device
        logger.info("[TODO] Load serialized TensorRT engine: %s", self.engine_path)
        if not self.engine_path.exists():
            logger.warning(
                "TRT engine file '%s' does not exist yet. Running in simulation mode.",
                self.engine_path,
            )

    def _allocate_buffers(self) -> None:
        """Allocate host and device pinned CUDA buffers for model input and output.

        TODO: Implement CUDA device buffer allocation via pycuda.driver:
            1. import pycuda.driver as cuda
            2. import pycuda.autoinit
            3. For each binding in self.engine:
                 - Query binding shape and datatype
                 - size = trt.volume(shape) * dtype.itemsize
                 - Allocate host page-locked memory: host_mem = cuda.pagelocked_empty(...)
                 - Allocate GPU device memory: device_mem = cuda.mem_alloc(size)
                 - Append to host/cuda inputs or outputs list
            4. self.stream = cuda.Stream()
        """
        # TODO: Replace with real PyCUDA buffer allocation on Jetson device
        logger.info("[TODO] Allocate GPU host/device memory buffers for input shape %s", self.input_shape)

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Execute synchronous or asynchronous inference on the input tensor.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W), float32.

        Returns:
            np.ndarray: Predicted segmentation mask or class logits (1, 4, H, W).

        TODO: Implement asynchronous GPU inference execution:
            1. np.copyto(self.host_inputs[0], input_tensor.ravel())
            2. cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            3. self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            4. cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            5. self.stream.synchronize()
            6. Reshape output buffer to (1, num_classes, H, W)
        """
        # SKELETON / SIMULATION: Generate dummy prediction mask for testing pipeline
        # (Replace with real TRT inference output)
        b, c, h, w = 1, len(CLASS_NAMES), self.input_shape[0], self.input_shape[1]
        dummy_logits = np.zeros((b, c, h, w), dtype=np.float32)

        # Simulate synthetic segmentation: mostly walkable ground (class 0)
        dummy_logits[:, 0, :, :] = 1.0
        # Add simulated non-walkable road boundary (class 1)
        dummy_logits[:, 1, : int(h * 0.4), :] = 2.0
        return dummy_logits


# ==============================================================================
# Frame Preprocessing & Postprocessing
# ==============================================================================
def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Preprocess raw BGR camera frame for deep learning segmentation model.

    Standard Cityscapes / ImageNet normalization pipeline:
    1. Resize to network input dimensions.
    2. Convert color space BGR -> RGB.
    3. Normalize pixel values with ImageNet mean and standard deviation.
    4. Transpose from HWC to NCHW format.

    Args:
        frame: Original camera frame (H, W, 3) in BGR uint8.
        target_size: Target resolution (Height, Width) for the model.

    Returns:
        np.ndarray: Batch tensor of shape (1, 3, H, W) in float32.
    """
    # 1. Resize
    resized = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # 2. Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 3. Normalize with ImageNet mean & std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std

    # 4. Transpose to (1, 3, H, W)
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0)
    return np.ascontiguousarray(tensor, dtype=np.float32)


def postprocess_mask(
    logits: np.ndarray,
    orig_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert model logits to a class prediction map and a colored overlay.

    Args:
        logits: Output tensor of shape (1, C, H, W) from model.
        orig_shape: Original camera frame resolution (Height, Width).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - class_map: 2D array (orig_H, orig_W) with predicted class IDs (0..3).
            - color_mask: 3D BGR image (orig_H, orig_W, 3) with color-coded classes.
    """
    # Extract prediction map: argmax over class channels
    class_map_small = np.argmax(logits[0], axis=0).astype(np.uint8)

    # Resize back to original frame dimensions
    class_map = cv2.resize(
        class_map_small,
        (orig_shape[1], orig_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Apply color map
    color_mask = COLOR_MAP[class_map]
    return class_map, color_mask


# ==============================================================================
# Walking Zone Obstacle Detection & Decision Logic
# ==============================================================================
def analyze_walking_zone(
    class_map: np.ndarray,
    zone_top_pct: float = 0.50,
    zone_bottom_pct: float = 0.95,
    zone_left_pct: float = 0.30,
    zone_right_pct: float = 0.70,
) -> Tuple[Optional[str], dict, Tuple[int, int, int, int]]:
    """Analyze the designated central walking corridor for navigation hazards.

    Extracts a bounding region of interest (ROI) in front of the user/robot and
    computes the class pixel distribution to decide appropriate audio alerts.

    Args:
        class_map: 2D segmentation prediction map (H, W).
        zone_top_pct: Top boundary of walking corridor as fraction of height.
        zone_bottom_pct: Bottom boundary of walking corridor as fraction of height.
        zone_left_pct: Left boundary of walking corridor as fraction of width.
        zone_right_pct: Right boundary of walking corridor as fraction of width.

    Returns:
        Tuple[Optional[str], dict, Tuple[int, int, int, int]]:
            - alert_id: String alert identifier (from audio_alerts.ALERTS) or None.
            - ratios: Dictionary of class area percentages within the ROI.
            - roi_box: (x1, y1, x2, y2) pixel bounding coordinates for visualization.
    """
    h, w = class_map.shape
    y1 = int(h * zone_top_pct)
    y2 = int(h * zone_bottom_pct)
    x1 = int(w * zone_left_pct)
    x2 = int(w * zone_right_pct)

    roi = class_map[y1:y2, x1:x2]
    total_pixels = roi.size

    if total_pixels == 0:
        return None, {}, (x1, y1, x2, y2)

    # Calculate class percentages
    ratios = {
        name: float(np.sum(roi == idx) / total_pixels)
        for idx, name in enumerate(CLASS_NAMES)
    }

    # Decision logic for audio alerts:
    alert_id = None
    if ratios["Dynamic Obstacle"] > 0.15:
        # Dynamic obstacles (pedestrians, vehicles, moving hazards)
        alert_id = "obstacle_ahead"
    elif ratios["Structural Obstacle"] > 0.20:
        # Static physical hazards (poles, barriers, stairs)
        alert_id = "stairs_ahead"
    elif ratios["Non-walkable"] > 0.60:
        # Entire path ahead is off-road, barrier, or impassable
        alert_id = "blocked_path"
    elif ratios["Walkable"] > 0.70:
        # Clean safe pathway detected
        alert_id = "path_clear"

    return alert_id, ratios, (x1, y1, x2, y2)


# ==============================================================================
# Main Live Inference Pipeline
# ==============================================================================
def run_live_inference(
    engine_path: str,
    camera_id: int = 0,
    lang: str = "en",
    audio_dir: str = "audio",
    input_size: Tuple[int, int] = (512, 512),
    alpha: float = 0.4,
) -> None:
    """Run the live video capture, TensorRT inference, and audio alert loop.

    Args:
        engine_path: Path to serialized TensorRT .engine file.
        camera_id: Device index for camera capture.
        lang: Audio alert language ('en' or 'hi').
        audio_dir: Path to base audio folder.
        input_size: Neural network spatial input dimensions (Height, Width).
        alpha: Overlay blending transparency weight (0.0 - 1.0).
    """
    # 1. Initialize Audio Alert System
    logger.info("Starting Audio Alert System (Language: %s)...", lang)
    alert_system = AudioAlertSystem(audio_dir=audio_dir, lang=lang, cooldown=4.0)
    alert_system.start()

    # 2. Initialize TensorRT Model Wrapper
    logger.info("Initializing TensorRT model wrapper...")
    trt_infer = TensorRTInference(engine_path=engine_path, input_shape=input_size)

    # 3. Open Camera Feed via GStreamer or OpenCV fallback
    logger.info("Opening camera device %d via GStreamer...", camera_id)
    pipeline_str = build_gstreamer_pipeline_usb(device_id=camera_id, width=1280, height=720, framerate=30)
    cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        logger.warning("GStreamer pipeline failed to open. Falling back to default cv2.VideoCapture(%d)...", camera_id)
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.error("Failed to open camera device %d. Exiting.", camera_id)
        alert_system.stop()
        sys.exit(1)

    logger.info("Camera successfully opened. Entering live processing loop. Press 'q' or 'ESC' to quit.")

    # FPS Calculation variables
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to receive frame from camera stream.")
                time.sleep(0.01)
                continue

            orig_h, orig_w = frame.shape[:2]

            # 4. Preprocess frame
            input_tensor = preprocess_frame(frame, target_size=input_size)

            # 5. TensorRT Inference
            logits = trt_infer.infer(input_tensor)

            # 6. Postprocess segmentation output & create colored overlay
            class_map, color_mask = postprocess_mask(logits, orig_shape=(orig_h, orig_w))
            blended = cv2.addWeighted(frame, 1.0 - alpha, color_mask, alpha, 0.0)

            # 7. Analyze Walking Zone ROI for obstacles
            alert_id, ratios, roi_box = analyze_walking_zone(class_map)
            x1, y1, x2, y2 = roi_box

            # 8. Trigger Audio Alert if hazard detected
            if alert_id is not None:
                alert_system.play(alert_id)

            # 9. Calculate and display FPS
            frame_count += 1
            now = time.time()
            if now - fps_start_time >= 0.5:
                fps = frame_count / (now - fps_start_time)
                frame_count = 0
                fps_start_time = now

            # 10. Draw UI / HUD Annotations
            # Draw walking zone corridor box
            box_color = (0, 0, 255) if alert_id and "obstacle" in alert_id else (0, 255, 0)
            cv2.rectangle(blended, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                blended,
                "Walking Corridor",
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
            )

            # Draw HUD status header
            hud_bg = blended.copy()
            cv2.rectangle(hud_bg, (0, 0), (orig_w, 45), (20, 20, 20), -1)
            blended = cv2.addWeighted(blended, 0.7, hud_bg, 0.3, 0)

            status_text = f"FPS: {fps:5.1f} | Alert: {alert_id or 'None'} | Lang: {lang.upper()}"
            cv2.putText(
                blended,
                status_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            # 11. Display rendered frame
            cv2.imshow("Jetson Orin Nano - Assistive Navigation", blended)

            # 12. Handle keyboard interrupts
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                logger.info("User requested exit.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        # Clean resource teardown
        logger.info("Releasing camera and stopping audio system...")
        cap.release()
        cv2.destroyAllWindows()
        alert_system.stop()
        logger.info("Live inference pipeline terminated cleanly.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command line arguments for live inference."""
    parser = argparse.ArgumentParser(
        description="Real-time Semantic Segmentation & Audio Alerts on Jetson Orin Nano",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="Path to optimized TensorRT engine file (.engine)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="V4L2 camera device index (e.g., 0 for /dev/video0)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "hi"],
        default="en",
        help="Audio alert spoken language ('en' for English, 'hi' for Hindi)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="audio",
        help="Root directory containing language audio clip folders",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=512,
        help="Model input width in pixels",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=512,
        help="Model input height in pixels",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_live_inference(
        engine_path=args.engine,
        camera_id=args.camera,
        lang=args.lang,
        audio_dir=args.audio_dir,
        input_size=(args.input_height, args.input_width),
    )
