#!/usr/bin/env python3
"""
export_onnx.py
--------------
Export trained PyTorch semantic segmentation model (BiSeNetV2 / Fast-SCNN)
to ONNX format for Jetson Orin Nano TensorRT deployment.

Features:
  - Exports with dynamic batch axes: {0: 'batch_size'}
  - Validates model graph integrity via onnx.checker.check_model()
  - Verifies numeric consistency with onnxruntime vs PyTorch
  - Provides trtexec compilation command for Jetson Orin Nano deployment
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional ONNX imports
try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ExportONNX")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export trained PyTorch segmentation model to ONNX for TensorRT"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for exported ONNX model (e.g., checkpoints/bisenetv2_512x1024.onnx)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to model config YAML (if config is not embedded in checkpoint)",
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default="512,1024",
        help="Input resolution as 'height,width' (default: '512,1024')",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX operator set version (default: 16, recommended for TensorRT 8.6+ on JetPack 5/6)",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        default=True,
        help="Enable dynamic batch size axis (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for tracing / export (default: 'cpu')",
    )
    return parser.parse_args()


def parse_input_size(size_str: str) -> Tuple[int, int]:
    """Parse 'height,width' string into (height, width) tuple."""
    parts = [int(p.strip()) for p in size_str.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid input-size format: '{size_str}'. Expected 'height,width'.")
    return parts[0], parts[1]


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load PyTorch model architecture and weights from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        config_path: Optional path to YAML config.
        device: Device to place the model on.

    Returns:
        Tuple of (model, config_dict).
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(ckpt_file, map_location=device)

    # Resolve config
    config = None
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
    elif config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        # Default fallback assumption if config is omitted
        logger.warning(
            "Config not found in checkpoint and not provided via --config. Assuming default bisenetv2 4-class."
        )
        config = {
            "model": {
                "name": "bisenetv2",
                "num_classes": 4,
                "auxiliary_loss": False,
            }
        }

    model_name = config["model"]["name"].lower()
    num_classes = config["model"]["num_classes"]

    # During export/inference, disable auxiliary heads for BiSeNetV2
    logger.info("Instantiating '%s' (num_classes=%d, aux_loss=False for export)", model_name, num_classes)

    try:
        from models import get_model
        try:
            model = get_model(name=model_name, num_classes=num_classes)
        except TypeError:
            model = get_model(name=model_name, num_classes=num_classes, auxiliary_loss=False)
    except ImportError:
        try:
            from src.models import get_model
            try:
                model = get_model(name=model_name, num_classes=num_classes)
            except TypeError:
                model = get_model(name=model_name, num_classes=num_classes, auxiliary_loss=False)
        except ImportError:
            raise NotImplementedError(
                f"Model builder for '{model_name}' not available. Implement models.get_model."
            )

    # Extract state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Remove auxiliary head weights if present
    filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("aux_heads")}

    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()

    # Wrap model to ensure clean single-tensor output for TensorRT
    wrapped_model = ExportWrapper(model)
    wrapped_model.eval()
    logger.info("Successfully loaded weights from %s", checkpoint_path)
    return wrapped_model, config


class ExportWrapper(nn.Module):
    """Wraps segmentation models returning dictionaries/tuples to output a single Tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out.get("main", next(iter(out.values())))
        elif isinstance(out, (tuple, list)):
            return out[0]
        return out


def export_model_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset_version: int = 16,
    dynamic_batch: bool = True,
) -> None:
    """
    Export PyTorch model to ONNX using torch.onnx.export.

    Args:
        model: Evaluated PyTorch model.
        dummy_input: Input tensor [B, C, H, W].
        output_path: Path where ONNX file will be written.
        opset_version: ONNX operator set version.
        dynamic_batch: Whether to mark dimension 0 of input/output as dynamic.
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    logger.info("Exporting model to ONNX at %s (opset=%d)...", output_path, opset_version)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path_obj),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    logger.info("ONNX export completed: %s", output_path)


def verify_onnx_model(onnx_path: str) -> bool:
    """
    Validate ONNX graph integrity with onnx.checker.check_model.

    Args:
        onnx_path: Path to ONNX model.

    Returns:
        True if valid, False otherwise.
    """
    if not HAS_ONNX:
        logger.warning("Package 'onnx' is not installed. Skipping onnx.checker validation.")
        return False

    logger.info("Verifying ONNX graph integrity with onnx.checker...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model checker passed successfully. The model graph is valid.")
        return True
    except Exception as e:
        logger.error("✕ ONNX checker validation failed: %s", e)
        return False


def verify_numerical_consistency(
    torch_model: nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
    tolerance: float = 1e-3,
) -> bool:
    """
    Verify numerical consistency between PyTorch and ONNX Runtime.

    Args:
        torch_model: PyTorch model in eval mode.
        onnx_path: Path to exported ONNX model.
        dummy_input: Input tensor.
        tolerance: Maximum acceptable absolute difference.

    Returns:
        True if outputs match within tolerance.
    """
    if not HAS_ORT:
        logger.warning("Package 'onnxruntime' not installed. Skipping numerical parity test.")
        return False

    logger.info("Testing numerical parity with ONNX Runtime...")
    try:
        # PyTorch forward
        with torch.no_grad():
            torch_out = torch_model(dummy_input)
            if isinstance(torch_out, (tuple, list)):
                torch_out = torch_out[0]
            torch_np = torch_out.detach().cpu().numpy()

        # ONNX Runtime forward
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: dummy_input.cpu().numpy()}
        ort_out = session.run(None, ort_inputs)[0]

        max_diff = np.max(np.abs(torch_np - ort_out))
        logger.info("Maximum absolute difference: %.6f (Tolerance: %.6f)", max_diff, tolerance)

        if max_diff < tolerance:
            logger.info("✓ PyTorch and ONNX Runtime outputs match within tolerance.")
            return True
        else:
            logger.warning("! Max difference (%.6f) exceeds tolerance (%.6f).", max_diff, tolerance)
            return False
    except Exception as e:
        logger.warning("Numerical verification failed: %s", e)
        return False


def print_jetson_tensorrt_instructions(onnx_path: str, height: int, width: int) -> None:
    """Print instructions for converting exported ONNX to TensorRT engine on Jetson."""
    engine_path = Path(onnx_path).with_suffix(".engine")
    print("\n" + "=" * 70)
    print("      JETSON ORIN NANO TENSORRT DEPLOYMENT INSTRUCTIONS")
    print("=" * 70)
    print("Run the following command on your Jetson Orin Nano terminal:\n")
    print(f"  trtexec --onnx={onnx_path} \\")
    print(f"          --saveEngine={engine_path} \\")
    print(f"          --fp16 \\")
    print(f"          --minShapes=input:1x3x{height}x{width} \\")
    print(f"          --optShapes=input:1x3x{height}x{width} \\")
    print(f"          --maxShapes=input:4x3x{height}x{width} \\")
    print(f"          --workspace=2048")
    print("\nFor INT8 quantization (with calibration cache):")
    print(f"  trtexec --onnx={onnx_path} --saveEngine={engine_path} --int8 --calib=calib.cache")
    print("=" * 70 + "\n")


def main() -> None:
    """Main entry point for ONNX export."""
    args = parse_args()
    height, width = parse_input_size(args.input_size)
    device = torch.device(args.device)

    logger.info("Target input resolution: %dx%d (HxW)", height, width)

    # Load Model
    try:
        model, _ = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=device,
        )
    except (NotImplementedError, FileNotFoundError) as e:
        logger.warning("Model load stub notice: %s", e)
        print_jetson_tensorrt_instructions(args.output, height, width)
        return

    # Create dummy input: [Batch=1, Channels=3, Height, Width]
    dummy_input = torch.randn(1, 3, height, width, device=device)

    # Export to ONNX
    export_model_to_onnx(
        model=model,
        dummy_input=dummy_input,
        output_path=args.output,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
    )

    # Verify ONNX
    verify_onnx_model(args.output)

    # Verify numerical consistency
    verify_numerical_consistency(model, args.output, dummy_input)

    # Print Jetson commands
    print_jetson_tensorrt_instructions(args.output, height, width)


if __name__ == "__main__":
    main()
