# Limitations & Ethical Considerations

**Project:** Walkability Semantic Segmentation — Capstone Project  
**Institution:** Ramdeobaba University, Nagpur — Dept. of Electronics Engineering  
**Authors:** Divyansh Gupta, Arnav Pangul, Gauri Thakre, Dhruv Runwal  

---

> [!CAUTION]
> **IMPORTANT SAFETY DISCLAIMER: RESEARCH PROTOTYPE ONLY**  
> This software and hardware implementation is an academic research prototype developed solely for educational and experimental purposes. **IT IS NOT A CERTIFIED MEDICAL, LIFE-SAFETY, MOBILITY, OR INDUSTRIAL SAFETY DEVICE.**  
> Under no circumstances should this system be deployed or relied upon as a primary or sole navigation aid for visually impaired individuals, persons with limited mobility, or any critical path navigation. Users must retain their primary conventional mobility tools (such as certified white canes or guide dogs) and rely on human situational awareness at all times.

---

## 1. Scope & Intended Use

### 1.1 Intended Purpose
The primary objective of this project is to explore real-time edge-based semantic segmentation (using BiSeNetV2 and Fast-SCNN architectures compiled with NVIDIA TensorRT) for surface walkability understanding and obstacle presence detection. The system provides real-time auditory notifications intended to supplement, rather than replace, human sensory perception in controlled or semi-controlled walking environments (such as university campus footpaths, sidewalks, and open courtyards).

### 1.2 Operational Domain & Boundaries
- **Design Target:** Daylight, outdoor and transition pedestrian environments under standard visibility conditions.
- **Out of Scope:** High-speed vehicular navigation, indoor navigation without natural lighting, subterranean tunnels, dense industrial machinery zones, steep off-road hiking trails, emergency evacuation guidance.

---

## 2. Known Failure Modes

Understanding failure cases is crucial for academic evaluation and real-world safety analysis:

1. **Adverse Lighting & Glare:**
   - **Direct Sunlight & Glare:** Intense solar glare directly facing the camera sensor causes sensor saturation (blooming) and washouts, obliterating path boundary gradients.
   - **Deep Shadows:** High-contrast shadows cast by tall buildings or dense foliage can cause misclassification of dark sidewalk zones as non-walkable road or structural barriers.
   - **Night / Ultra Low-Light:** In the absence of an active infrared illuminator, low signal-to-noise ratio in low ambient light leads to heavy pixel noise and drastic confidence drops.

2. **Negative Obstacles & Drop-offs:**
   - Monocular RGB semantic segmentation lacks explicit 3D geometric depth perception.
   - Descending staircases, drop-off curbs, uncovered drainage trenches, and sudden pits ("negative obstacles") may visually resemble standard flat surfaces from a frontal perspective and fail to trigger timely alerts.

3. **Transparent & Specular Surfaces:**
   - Transparent glass doors, storefronts, and full-length glass barriers can be erroneously categorized as open walkable passages.
   - Polished wet marble, ceramic tiles, or large reflective puddles can mirror sky or overhead structures, confusing edge delineation.

4. **Crowd Occlusion:**
   - In dense pedestrian crowds, the ground plane is almost entirely occluded by nearby people, hindering walkable corridor identification.

5. **Motion Blur & Dynamic Velocity Mismatch:**
   - Rapid head or body movements can cause severe motion blur that degrades segmentation quality.
   - Fast-approaching vehicles or speeding bicycles may close the distance faster than the model inference and audio alert cycle can react.

---

## 3. Privacy Considerations

The deployment of body-worn or handheld cameras in public and semi-public environments introduces important privacy considerations:

- **Edge Processing & Zero Persistent Storage:**  
  The system performs on-device real-time inference using NVIDIA Jetson Orin Nano. Video frames are captured directly into volatile system memory (RAM/VRAM), processed for semantic segmentation, and immediately discarded. **No continuous raw video or audio footage is stored to persistent flash storage or transmitted over the network.**
- **De-identification in Datasets:**  
  For campus training data collection, explicit measures are taken to blur human faces and motor vehicle license plates prior to storing or distributing datasets.
- **Bystander Rights & Ethical Consent:**  
  Data collection on campus grounds complies with institutional ethical guidelines. Data collectors do not target private residences, restrooms, or restricted facilities.

---

## 4. Dataset Bias & Geographic Domain Shift

Semantic segmentation models are inherently susceptible to domain distribution shifts:

- **Cityscapes Dataset Bias:**  
  The Cityscapes dataset consists of German and Western European urban driving scenes characterized by standardized lane markers, uniform asphalt textures, distinct concrete curbs, and disciplined traffic flow. Models trained purely on Cityscapes struggle when deployed in unstructured walking environments.
- **Indian Driving Dataset (IDD) Realities:**  
  IDD provides vital representation of unstructured road edges, diverse motorized and non-motorized vehicles, and unpaved verges common in South Asian environments. However, IDD is primarily recorded from a vehicle dashboard vantage point (~1.6m – 1.8m height along roadways) rather than a pedestrian footpath perspective.
- **Campus Dataset Representation:**  
  While the custom Campus dataset bridges the gap between road driving views and pedestrian walking corridors, it is gathered primarily on a single university campus. Features such as specific paver stone patterns, brick styles, or local regional foliage may limit generalization to drastically different environments (e.g., rural unpaved tracks or snow-covered paths).

---

## 5. Latency & Safety Budget

In assistive navigation, system latency directly correlates with physical travel distance before an alert is delivered.

- **Human Walking Velocity:** Average pedestrian walking speed is approximately $1.2\text{ m/s}$ to $1.4\text{ m/s}$.
- **Computational Pipeline Latency:**
  - Image Acquisition & Preprocessing: $\sim 8\text{ ms} - 12\text{ ms}$
  - TensorRT FP16 Model Inference (Jetson Orin Nano): $\sim 15\text{ ms} - 25\text{ ms}$
  - Post-processing & Heuristic Threat Analysis: $\sim 5\text{ ms} - 10\text{ ms}$
  - Total Visual Processing Latency: $\sim 30\text{ ms} - 45\text{ ms}$ ($\ge 22 - 30\text{ FPS}$)
- **Audio Feedback Latency:**
  - Synthesized speech or auditory cue latency: $\sim 150\text{ ms} - 300\text{ ms}$
- **Total Safety Margin:**
  At $1.4\text{ m/s}$, an alert delayed by $350\text{ ms}$ results in the user moving forward by approximately $0.49\text{ m}$ before perceiving the cue. Consequently, the detection zone must maintain an active forward lookahead buffer of at least **$2.0\text{ m} - 3.0\text{ m}$** to ensure adequate reaction time.

---

## 6. Disclaimer

THE SOFTWARE AND CODE IN THIS REPOSITORY ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS, RAMDEOBABA UNIVERSITY, OR CONTRIBUTORS BE LIABLE FOR ANY CLAIM, DAMAGES, ACCIDENT, INJURY, OR OTHER LIABILITY ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
