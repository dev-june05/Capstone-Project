# Annotation Protocol for Campus Dataset

**Project:** Walkability Semantic Segmentation — Capstone Project  
**Institution:** Ramdeobaba University, Nagpur — Dept. of Electronics Engineering  
**Tool:** [Roboflow](https://roboflow.com)  

---

## 1. Overview & Objective

The Campus Dataset is curated to capture the real-world operational domain of an assistive walking aid within an Indian university campus. Campus pathways present distinct visual challenges compared to public roadways, including unstructured walkway boundaries, mixed paved/unpaved surfaces, pedestrians, stray animals, cyclists, overhanging foliage, and temporary construction or maintenance hazards.

This protocol standardizes the image collection, pixel-level semantic annotation, and quality assurance workflows across all annotators to ensure ground-truth consistency.

---

## 2. Target Semantic Classes

Every pixel in an image must be assigned to exactly one of the following four semantic classes or labeled as `Void / Ignore` if outside the field of navigation relevance (e.g., sky, extreme background beyond 30 meters).

| Class ID | Class Name | Description | Key Examples | Color Code (RGB) |
|---|---|---|---|---|
| **0** | **Walkable** | Safe, contiguous ground surfaces intended or safe for pedestrian walking. Must be free of physical barriers. | Sidewalks, pedestrian footpaths, designated walkways, pedestrian crossings, ramps, safe asphalt paths. | `[0, 255, 0]` (Green) |
| **1** | **Non-walkable** | Surfaces that are either impassable, legally restricted, hazardous, or not intended for pedestrian foot traffic. | Motor vehicle roads (carriageways), lawns, gardens, dirt verges, water bodies, curbs > 10 cm, closed entrance doors, steep slopes. | `[255, 0, 0]` (Red) |
| **2** | **Dynamic Obstacle** | Objects capable of self-movement or actively changing position in real-time, posing transient collision risks. | Pedestrians, moving cyclists, moving motorized vehicles (cars, e-rickshaws, bikes), moving animals (dogs, cattle). | `[255, 255, 0]` (Yellow) |
| **3** | **Structural Obstacle** | Fixed, static, or semi-permanent physical structures and objects that obstruct pedestrian navigation or represent trip/impact hazards. | Light poles, bollards, fences, walls, tree trunks, signboards, parked bicycles/motorcycles, potholes, construction barriers. | `[0, 0, 255]` (Blue) |

---

## 3. Edge Case Rules

To prevent ambiguity when delineating complex urban and campus environments, all annotators must follow these authoritative edge case rules:

| Surface / Object Type | Annotated Class | Classification Logic & Annotation Guidance |
|---|---|---|
| **Speed bump** | **Walkable** | Traversable ground elevation. While a mild vertical variance exists, pedestrians can step over or walk directly on it without obstruction. |
| **Pothole** | **Structural** | Severe depression and trip/fall hazard. Annotate the depression basin as a Structural Obstacle to alert the user to step around it. |
| **Low curb (< 10 cm)** | **Walkable** | Low step height easily negotiated by normal walking gait without blocking passage. |
| **High curb (≥ 10 cm)** | **Structural** | Significant step height or raised divider that poses a tripping or blocking barrier. |
| **Cracked tile** | **Walkable** | Surface imperfection; as long as the ground plane remains continuous and traversable, it remains Walkable. |
| **Wet floor / puddle** | **Walkable** | Continuous navigable pathway surface. Slippery conditions do not transform the ground plane into an impassable boundary. |
| **Open door** | **Walkable** | The open doorway floor threshold allows direct safe passage and is annotated as Walkable. |
| **Closed door** | **Non-walkable** | Physical barrier blocking forward progression through a doorway or entrance. |
| **Parked bicycle** | **Structural** | Stationary, non-moving physical barrier obstructing the pedestrian walkway. |
| **Moving bicycle** | **Dynamic** | Active motion obstacle with variable trajectory and velocity. |
| **Tree trunk** | **Structural** | Ground-rooted solid obstacle occupying ground space. |
| **Overhanging branch** | **Non-walkable** | Elevated obstacle protruding into walking envelope overhead; mark the projected impassable zone as Non-walkable. |

### Supplementary Delineation Rules

- **Shared Pedestrian-Vehicle Paths:** If pedestrians share a wide pathway with slow-moving campus vehicles and no distinct sidewalk exists, mark the clear navigable surface currently free of vehicles as **Walkable**, while any moving vehicles are strictly marked as **Dynamic Obstacle**.
- **Shadows & Occlusions:** Do not annotate shadows as boundaries. If a sidewalk is shaded by a tree or building, the underlying ground surface remains **Walkable**.
- **Bollards & Poles:** Trace the boundary tightly around the structural pole down to where it contacts the ground. Do not expand the polygon into the walking ground plane.
- **Door Thresholds:** For doorways, mark the visible ground threshold between doorposts as Walkable only if the door leaf is swung open by at least 70 degrees.

---

## 4. Image Collection Protocol

To mirror the real-world operational perspective of a body-worn assistive device on a pedestrian:

1. **Resolution & Format:**
   - Native capture: 1920 × 1080 (1080p Full HD) or 1280 × 720 (720p HD).
   - Aspect ratio: 16:9 widescreen format.
   - File formats: Uncompressed `.png` or high-quality `.jpg` (quality level ≥ 95%).

2. **Mounting Perspective & Camera Height:**
   - Camera Height: **1.3 m – 1.5 m** above ground level (chest-level mount or neck-worn harness).
   - Camera Pitch: Slight downward pitch angle of **10° – 15°** below the horizon to ensure the immediate forward ground path (0.5 m to 6 m) occupies the lower 60% of the field of view.
   - Stability: Use optical image stabilization or a handheld gimbal to minimize severe motion blur.

3. **Temporal & Illumination Distribution:**
   Images must be captured across three distinct daylight periods to provide variance in illumination, shadow length, and contrast:
   - **Morning (08:00 – 10:30):** Low-angle sunlight, long directional shadows, high specular reflections on pavers.
   - **Midday (12:00 – 14:30):** High overhead solar angle, minimal shadows, high brightness, washed-out highlights.
   - **Late Afternoon / Golden Hour (16:30 – 18:30):** Warm color temperature, transitional lighting, rapid ambient decay.

4. **Environmental Conditions:**
   - Include diverse path types: paved interlocking tiles, asphalt roads, concrete ramps, corridors, courtyard transitions.
   - Include varied crowd densities: solitary pathways, intermittent pedestrian traffic, and group congestion.

---

## 5. Quality Check Procedure

To guarantee annotation reliability and minimize label noise before model training:

1. **Annotation Environment:**
   - Platform: **Roboflow** web interface.
   - Annotators utilize polygon segmentation tools with zoom factors of at least 300% on boundary edges.

2. **Dual-Annotator Inter-Rater Reliability Test:**
   - **Sample Cohort:** 30 randomly selected images from each collection batch are assigned independently to two annotators (Annotator A and Annotator B) without viewing each other's masks.
   - **Evaluation Metric:** Class-wise and Mean Intersection over Union (mIoU) calculated across all four target classes:
     $$\text{IoU}_c = \frac{|P_A^{(c)} \cap P_B^{(c)}|}{|P_A^{(c)} \cup P_B^{(c)}|}$$
     $$\text{mIoU} = \frac{1}{4} \sum_{c=0}^{3} \text{IoU}_c$$
   - **Acceptance Threshold:** The pair must achieve a mean agreement threshold of **mIoU ≥ 85%**.

3. **Discrepancy Resolution & Calibration:**
   - If agreement falls below 85%:
     1. Discrepant regions are extracted and highlighted.
     2. A consensus review is conducted between the annotators and the Project Lead.
     3. Systematic disagreements (e.g., curb height interpretation or tree canopy projections) are reconciled by referencing Section 3.
     4. Annotators re-annotate the failed batch until the 85% threshold is consistently surpassed.
