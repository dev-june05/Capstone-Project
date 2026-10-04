#!/usr/bin/env bash
# ==============================================================================
# convert_trt.sh
#
# Convert an ONNX semantic segmentation model (BiSeNetV2 / Fast-SCNN)
# to a TensorRT FP16 optimized engine for deployment on NVIDIA Jetson Orin Nano.
#
# Usage:
#   chmod +x convert_trt.sh
#   ./convert_trt.sh <path_to_onnx_model> [output_engine_path] [workspace_mb]
#
# Examples:
#   ./convert_trt.sh checkpoints/bisenetv2_cityscapes.onnx
#   ./convert_trt.sh checkpoints/fast_scnn_idd.onnx checkpoints/fast_scnn.engine 2048
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Parameter Validation & Argument Parsing
# ------------------------------------------------------------------------------
if [ "$#" -lt 1 ]; then
    echo "======================================================================"
    echo "Error: Missing required ONNX model argument."
    echo "Usage: $0 <path_to_onnx_model> [output_engine_path] [workspace_mb]"
    echo ""
    echo "Arguments:"
    echo "  <path_to_onnx_model>  (Required) Path to the input .onnx model."
    echo "  [output_engine_path]  (Optional) Destination path for .engine file."
    echo "                        Defaults to <path_to_onnx_model>.engine"
    echo "  [workspace_mb]        (Optional) Max scratch workspace in MB."
    echo "                        Defaults to 2048 MB."
    echo "======================================================================"
    exit 1
fi

ONNX_FILE="$1"
ENGINE_FILE="${2:-${ONNX_FILE%.onnx}.engine}"
WORKSPACE_MB="${3:-2048}"

# Verify input file exists
if [ ! -f "$ONNX_FILE" ]; then
    echo "Error: ONNX file not found at '$ONNX_FILE'"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Locate trtexec Binary on JetPack / Jetson System
# ------------------------------------------------------------------------------
TRTEXEC_BIN="trtexec"

if ! command -v "$TRTEXEC_BIN" &> /dev/null; then
    # Standard JetPack path on Ubuntu / Jetson Linux
    JETPACK_TRTEXEC="/usr/src/tensorrt/bin/trtexec"
    if [ -x "$JETPACK_TRTEXEC" ]; then
        TRTEXEC_BIN="$JETPACK_TRTEXEC"
    else
        echo "Error: 'trtexec' command not found in PATH or at '$JETPACK_TRTEXEC'."
        echo "Please ensure NVIDIA TensorRT and JetPack SDK are installed."
        exit 1
    fi
fi

# Ensure output directory exists
mkdir -p "$(dirname "$ENGINE_FILE")"

# ------------------------------------------------------------------------------
# 3. Explanation of TensorRT CLI Flags
# ------------------------------------------------------------------------------
# --onnx=<path>
#     Specifies the path to the input ONNX model to be parsed and compiled.
#
# --saveEngine=<path>
#     Specifies the output destination file where the serialized TensorRT
#     binary plan (engine) will be saved for runtime deployment.
#
# --fp16
#     Enables FP16 (half-precision 16-bit floating point) mode.
#     On NVIDIA Jetson Orin Nano (Ampere GPU architecture with Tensor Cores),
#     FP16 provides ~2x throughput acceleration and halves GPU VRAM bandwidth,
#     while preserving visual segmentation quality.
#
# --workspace=<size_in_mb> / --memPoolSize=workspace:<size_in_mib>
#     Allocates the maximum scratch memory size (in megabytes) available to
#     TensorRT algorithms and tactic selection during engine building.
#     2048 MB (2 GB) is recommended for Orin Nano (4GB / 8GB unified memory).
#
# Additional flags that can be appended if needed:
#   --verbose:
#       Enables detailed logging of graph optimization, fusion passes, and tactics.
#   --profilingVerbosity=detailed:
#       Enables detailed per-layer latency breakdowns for NVTX and profiling.
#   --avgRuns=100:
#       Averages timing across 100 benchmark iterations after building.
#   --dumpProfile:
#       Prints per-layer latency and performance metrics to stdout.
# ------------------------------------------------------------------------------

echo "======================================================================"
echo "          TensorRT Engine Conversion for Jetson Orin Nano             "
echo "======================================================================"
echo " Input ONNX Model : ${ONNX_FILE}"
echo " Output TRT Engine: ${ENGINE_FILE}"
echo " Workspace Limit  : ${WORKSPACE_MB} MB"
echo " trtexec Path     : ${TRTEXEC_BIN}"
echo " Precision        : FP16 (Ampere Tensor Cores enabled)"
echo "======================================================================"

# Determine TensorRT version syntax:
# TRT 8.6+ and TRT 10+ deprecate --workspace in favor of --memPoolSize=workspace:XMiB.
# We supply both or fallback depending on trtexec version help output.
WORKSPACE_FLAG="--workspace=${WORKSPACE_MB}"
if "$TRTEXEC_BIN" --help 2>&1 | grep -q "memPoolSize"; then
    WORKSPACE_FLAG="--memPoolSize=workspace:${WORKSPACE_MB}MiB"
fi

# ------------------------------------------------------------------------------
# 4. Execute trtexec Conversion
# ------------------------------------------------------------------------------
START_TIME=$(date +%s)

echo "[INFO] Starting TensorRT engine compilation. This may take several minutes..."
echo "[CMD]  ${TRTEXEC_BIN} --onnx=\"${ONNX_FILE}\" --saveEngine=\"${ENGINE_FILE}\" --fp16 ${WORKSPACE_FLAG}"

"$TRTEXEC_BIN" \
    --onnx="${ONNX_FILE}" \
    --saveEngine="${ENGINE_FILE}" \
    --fp16 \
    ${WORKSPACE_FLAG}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================================================"
echo "Conversion complete in ${ELAPSED} seconds!"
echo "Saved TensorRT engine: ${ENGINE_FILE}"
echo "File size: $(du -h "${ENGINE_FILE}" | cut -f1)"
echo "======================================================================"
