# Walkability Semantic Segmentation — Capstone Project

**Real-Time 4-Class Semantic Segmentation for Assisted Walking on Embedded Edge Hardware**

**Department of Electronics Engineering**  
**Ramdeobaba University, Nagpur**

---

## 👥 Project Team

- **Divyansh Gupta** — *Model Training Lead*
- **Arnav Pangul** — *Data Pipeline Lead*
- **Gauri Thakre** — *Jetson Deployment Lead*
- **Dhruv Runwal** — *UI & Audio Testing Lead*

---

## 📖 Overview

Visual navigation assistance for pedestrians requires low-latency, semantically rich scene understanding directly at the edge. This project develops a real-time, lightweight semantic segmentation system running on the **NVIDIA Jetson Orin Nano**, specifically engineered to identify safe traversable ground planes and flag hazards for assisted walking.

The system condenses high-cardinality urban datasets into a task-focused **4-class walkability taxonomy**:
1. **Walkable (ID: 0)** — Footpaths, sidewalks, pedestrian crossings, safe pathways.
2. **Non-walkable (ID: 1)** — Motor roadways, off-limit verges, lawns, bodies of water, closed boundaries.
3. **Dynamic Obstacle (ID: 2)** — Moving pedestrians, cyclists, motorized vehicles, and animals.
4. **Structural Obstacle (ID: 3)** — Fixed street furniture, poles, fences, walls, parked vehicles, high curbs ($\ge 10\text{ cm}$), and potholes.

We benchmark and optimize two leading real-time architectures — **BiSeNetV2** and **Fast-SCNN** — trained on a fusion of **Cityscapes** and the **Indian Driving Dataset (IDD)**, along with a custom annotated campus dataset. Trained models are exported to **ONNX** and compiled into high-efficiency **TensorRT** FP16 runtime engines, driving real-time spatial heuristics and directional audio alerts for users.

---

## 📁 Project Structure

```
Capstone Project/
├── configs/                        # YAML configuration files
│   ├── train_bisenetv2.yaml        # BiSeNetV2 training hyperparameters
│   ├── train_fastscnn.yaml         # Fast-SCNN training hyperparameters
│   └── jetson_config.yaml          # TensorRT inference & camera parameters
├── data/                           # Dataset storage (excluded from git via .gitignore)
│   ├── .gitkeep                    # Instructions for dataset download
│   ├── cityscapes/                 # Cityscapes leftImg8bit and gtFine
│   ├── idd/                        # IDD segmentation dataset
│   └── campus/                     # Collected campus images & Roboflow masks
├── docs/                           # Project documentation & guidelines
│   ├── annotation_protocol.md      # Ground-truth tagging rules & edge cases
│   └── ethics_limitations.md       # Limitations, safety boundaries & ethics
├── models/                         # PyTorch model definitions
│   ├── __init__.py
│   ├── bisenetv2.py                # BiSeNetV2 architecture (Detail + Semantic Branch)
│   ├── fast_scnn.py                # Fast-SCNN architecture (Learning to Downsample)
│   └── common.py                   # Shared conv blocks, heads, and loss functions
├── notebooks/                      # Exploratory data analysis & prototyping
│   └── .gitkeep
├── scripts/                        # Utility & operational scripts
│   ├── data_prep/
│   │   ├── remap_cityscapes.py     # Cityscapes 19-class to 4-class remapper
│   │   ├── remap_idd.py            # IDD 26-class to 4-class remapper
│   │   └── verify_labels.py        # Label verification & class distribution stats
│   ├── train.py                    # Model training pipeline with AMP & WandB
│   ├── evaluate.py                 # Evaluation script (mIoU, class IoU, confusion matrix)
│   ├── export_onnx.py              # PyTorch to ONNX export and graph validation
│   └── build_engine.py             # ONNX to TensorRT engine compilation script
├── deployment/                     # Embedded runtime on NVIDIA Jetson
│   ├── jetson_inference.py         # Real-time PyCUDA + TensorRT camera inference
│   ├── audio_alerts.py             # Spatial threat heuristics & audio feedback engine
│   └── benchmark.py                # Latency, FPS, and memory benchmarking
├── utils/                          # Common utility modules
│   ├── __init__.py
│   ├── dataset.py                  # PyTorch Dataset & DataLoader implementations
│   ├── metrics.py                  # Streamed IoU, mIoU, and confusion matrix metrics
│   ├── visualization.py            # Color mask blending and plotting utilities
│   └── logger.py                   # Logging & WandB interface
├── .gitignore                      # Git ignore rules for data, weights & artifacts
├── requirements.txt                # Python dependencies for development PC
├── requirements-jetson.txt         # Lightweight dependencies for Jetson Orin Nano
└── README.md                       # Project documentation
```

---

## 🌿 Branching Strategy

Our team utilizes a structured Git workflow to isolate development areas and facilitate continuous integration:

| Branch | Purpose & Ownership | Integration Rules |
|---|---|---|
| `main` | Production-ready, verified releases tested on the Jetson hardware. | Protected branch. Merges only via Pull Requests approved by team leads. |
| `dev` | Central integration branch for ongoing development. | PRs from feature branches must pass linting and baseline tests. |
| `arnav/data-pipeline` | Data downloading, cleaning, 4-class remapping, and augmentation pipeline. | Maintained by Arnav Pangul. |
| `divyansh/model-training` | Model architectures (BiSeNetV2, Fast-SCNN), losses, and training loops. | Maintained by Divyansh Gupta. |
| `gauri/jetson-deployment` | ONNX export, TensorRT engine compilation, FP16 tuning, Jetson runtime. | Maintained by Gauri Thakre. |
| `dhruv/ui-testing` | Audio alert engine, spatial hazard heuristics, edge benchmarking, and testing. | Maintained by Dhruv Runwal. |

---

## 🛠️ Setup & Installation

### 1. Workstation / Training PC Setup (CUDA-enabled GPU)

```bash
# Clone the repository
git clone https://github.com/YourOrg/walkability-segmentation.git
cd "Capstone Project"

# Create and activate a Python virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Upgrade pip and install development dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Embedded Target Setup (NVIDIA Jetson Orin Nano)

> [!NOTE]
> JetPack SDK (5.x or 6.x) provides pre-installed, hardware-accelerated builds of CUDA, cuDNN, TensorRT, PyTorch, and torchvision. Do not overwrite these with standard PyPI wheels.

```bash
# On Jetson Orin Nano terminal:
git clone https://github.com/YourOrg/walkability-segmentation.git
cd "Capstone Project"

# Install edge runtime helper packages
pip3 install -r requirements-jetson.txt
```

---

## 📊 Dataset Preparation

### 1. Download Datasets
- **Cityscapes:** Register and download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/). Extract into `data/cityscapes/`.
- **Indian Driving Dataset (IDD):** Download IDD Segmentation (Part 1 & 2) from [idd.insaan.iiit.ac.in](https://idd.insaan.iiit.ac.in/). Extract into `data/idd/`.
- **Campus Dataset:** Annotated images exported from Roboflow are placed in `data/campus/`.

### 2. Label Remapping to 4 Classes

Execute the remapping scripts to convert the 19 Cityscapes classes and 26 IDD classes into our standardized 4-class walkability scheme:

```bash
# Remap Cityscapes annotations
python scripts/data_prep/remap_cityscapes.py \
    --data_dir data/cityscapes \
    --output_dir data/processed/cityscapes

# Remap Indian Driving Dataset (IDD) annotations
python scripts/data_prep/remap_idd.py \
    --data_dir data/idd \
    --output_dir data/processed/idd

# Verify remapped masks and print class distribution statistics
python scripts/data_prep/verify_labels.py \
    --data_dir data/processed
```

---

## 🚀 Model Training

Train the dual-pathway real-time segmentation networks with mixed precision (AMP) and automatic checkpointing:

```bash
# Train BiSeNetV2 on combined datasets
python scripts/train.py --config configs/train_bisenetv2.yaml

# Train Fast-SCNN on combined datasets
python scripts/train.py --config configs/train_fastscnn.yaml

# Resume training from an interrupted checkpoint
python scripts/train.py --config configs/train_bisenetv2.yaml --resume checkpoints/bisenetv2_epoch40.pth
```

Key configurations in `configs/train_bisenetv2.yaml`:
- **Input resolution:** $512 \times 1024$ (or $512 \times 512$ for Fast-SCNN)
- **Batch size:** 8 or 16
- **Optimizer:** AdamW or SGD with polynomial learning rate scheduler
- **Loss:** CrossEntropyLoss + OHEM / Dice Loss weighted for class imbalance

---

## 📈 Evaluation

Evaluate validation sets and generate mean Intersection-over-Union (mIoU), per-class IoU, and confusion matrices:

```bash
# Evaluate BiSeNetV2 checkpoint
python scripts/evaluate.py \
    --config configs/train_bisenetv2.yaml \
    --checkpoint checkpoints/bisenetv2_best.pth \
    --split val

# Evaluate Fast-SCNN checkpoint
python scripts/evaluate.py \
    --config configs/train_fastscnn.yaml \
    --checkpoint checkpoints/fastscnn_best.pth \
    --split val
```

---

## 🔄 ONNX Export

Export PyTorch weights to an ONNX graph with static input dimensions tailored for TensorRT:

```bash
# Export BiSeNetV2 to ONNX
python scripts/export_onnx.py \
    --model bisenetv2 \
    --checkpoint checkpoints/bisenetv2_best.pth \
    --output checkpoints/bisenetv2.onnx \
    --input_shape 1 3 512 1024 \
    --opset 16

# Export Fast-SCNN to ONNX
python scripts/export_onnx.py \
    --model fast_scnn \
    --checkpoint checkpoints/fastscnn_best.pth \
    --output checkpoints/fastscnn.onnx \
    --input_shape 1 3 512 1024 \
    --opset 16
```

---

## 🌐 Cloud Web App & Browser Inference (Vercel)

We provide a **100% Client-Side Web Application** running the exported BiSeNetV2 model directly inside modern web browsers using **ONNX Runtime Web (WASM SIMD)**.

- **Zero-Backend Serverless Architecture**: The 24.4 MB ONNX model runs purely on the user's client hardware.
- **Real-Time 4-Class Visualization**: Instant color-coded overlay with opacity control.
- **Directional Audio Safety System**: Immediate sonar acoustic chimes and bilingual TTS speech prompts (*English & Hindi*) detecting near-field obstacles in the bottom 40% walking corridor.
- **Local Testing**:
  ```bash
  python -m http.server 3000 --directory web
  ```
  Open `http://localhost:3000` in Google Chrome or Microsoft Edge.

---

## 📊 Final Quantitative Benchmark Results (Epoch 30)

Evaluated on the full 500-image Cityscapes validation split (`512x1024` resolution):

| Metric | BiSeNetV2 Result | Target Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Mean IoU (mIoU)** | **69.38%** | $\ge 60.0\%$ | ✅ Exceeded (+9.38%) |
| **Walkable Path IoU** | **94.95%** | $\ge 85.0\%$ | ✅ Exceeded (+9.95%) |
| **Non-Walkable IoU** | **89.53%** | $\ge 80.0\%$ | ✅ Exceeded (+9.53%) |
| **Dynamic Obstacle IoU** | **72.89%** | $\ge 60.0\%$ | ✅ Exceeded (+12.89%) |
| **Structural Obstacle IoU** | **20.13%** | $\ge 20.0\%$ | ✅ Achieved |
| **Combined Obstacle Recall** | **75.98%** | $\ge 70.0\%$ | ✅ Exceeded (+5.98%) |
| **Missed Obstacles in Walking Zone** | **9.47%** | $\le 12.0\%$ | ✅ Safe (< 10%) |

---

## ⚡ Jetson Deployment (TensorRT)

Compile the ONNX computational graph into an FP16 TensorRT engine on the Jetson Orin Nano for maximum inference throughput:

### 1. Build TensorRT Engine

```bash
# Using trtexec command-line tool (recommended)
trtexec --onnx=checkpoints/bisenetv2.onnx \
        --saveEngine=checkpoints/bisenetv2_fp16.engine \
        --fp16 \
        --workspace=2048

# Alternatively, build engine using Python script
python scripts/build_engine.py \
    --onnx checkpoints/bisenetv2.onnx \
    --engine checkpoints/bisenetv2_fp16.engine \
    --precision fp16
```

### 2. Run Real-Time Camera Inference

Launch camera capture (CSI or USB webcam) and real-time segmentation on Jetson:

```bash
# Run real-time video inference from webcam / CSI camera
python deployment/jetson_inference.py \
    --engine checkpoints/bisenetv2_fp16.engine \
    --camera 0

# Benchmark latency and FPS over 500 frames
python deployment/benchmark.py \
    --engine checkpoints/bisenetv2_fp16.engine \
    --iterations 500
```

---

## 🔊 Audio Alerts & Navigation Heuristics

The system partitions the near-field region of interest (lower central third of the frame) into three spatial sectors: **Left**, **Center**, and **Right**.

- **Navigable Corridor Check:** If the center ground sector lacks sufficient Walkable pixels ($< 60\%$), an alert is triggered.
- **Obstacle Proximity Warning:** If a Dynamic Obstacle or Structural Obstacle breaches the critical danger threshold in the lower frame, an immediate directional audio cue is generated:
  - *"Caution: Obstacle ahead"*
  - *"Obstacle on your right — path clear left"*
  - *"Caution: Vehicle approaching"*

Run the pipeline with live audio feedback enabled:

```bash
# Run deployment inference with audio warnings enabled
python deployment/jetson_inference.py \
    --engine checkpoints/bisenetv2_fp16.engine \
    --camera 0 \
    --audio

# Standalone test of audio generation engine
python deployment/audio_alerts.py --test
```

---

## 🤝 Team & Work Distribution

| Team Member | Role | Core Responsibilities |
|---|---|---|
| **Divyansh Gupta** | Model Architecture & Training Lead | Model definitions (BiSeNetV2, Fast-SCNN), loss formulation (OHEM/Dice), training pipeline, hyperparameter tuning, validation. |
| **Arnav Pangul** | Data Engineering Lead | Cityscapes and IDD download/curation, 4-class label remapping pipelines, campus dataset annotation coordination, data augmentations. |
| **Gauri Thakre** | Embedded Systems Lead | JetPack setup, ONNX graph export, TensorRT FP16 engine compilation, PyCUDA inference harness, latency optimization on Jetson Orin Nano. |
| **Dhruv Runwal** | UI & Navigation Heuristics Lead | Spatial ROI hazard heuristics, text-to-speech audio feedback module, test benchmarking, verification suite, project demo integration. |

---

## 📜 Documentation Links

- [Annotation Protocol for Campus Dataset](file:///d:/Project/Capstone%20Project/docs/annotation_protocol.md)
- [Limitations & Ethical Considerations](file:///d:/Project/Capstone%20Project/docs/ethics_limitations.md)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📚 References

1. **BiSeNetV2:** Yu, C., Gao, C., Wang, J., Yu, G., Shen, C., & Sang, N. (2021). *BiSeNet V2: Bilateral Network with Guided Aggregation for Real-Time Semantic Segmentation.* International Journal of Computer Vision (IJCV), 129(11), 3051-3068.
2. **Fast-SCNN:** Poudel, R. P., Liwicki, S., & Cipolla, R. (2019). *Fast-SCNN: Fast Semantic Segmentation Network.* British Machine Vision Conference (BMVC).
3. **Cityscapes Dataset:** Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., & Schiele, B. (2016). *The Cityscapes Dataset for Semantic Urban Scene Understanding.* CVPR.
4. **Indian Driving Dataset (IDD):** Varma, G., Subramanian, A., Namboodiri, A., Chandraker, M., & Jawahar, C. V. (2019). *IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments.* WACV.
5. **NVIDIA TensorRT:** *NVIDIA TensorRT Developer Guide & Jetson Orin Nano Technical Reference.* [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt).
