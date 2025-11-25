* * * * *

📁 SportsVision+ : Master Architecture & Implementation Guide
=============================================================

Project Goal: End-to-end real-time football analytics system.

Hardware Constraint: Single Laptop GPU (RTX 3050/3060/4050).

Core Strategy: Asynchronous Level-of-Detail (LOD) Pipeline.

Data Source: SoccerNet (Tracking) for training, generic broadcast video for inference.

* * * * *

1\. Repository Structure (Frozen)
---------------------------------

Create this exact folder structure.

Plaintext

```
sportsvision-plus/
├── data/
│   ├── soccernet/           # Place raw downloaded SoccerNet data here
│   ├── processed/           # Output of your parser (YOLO format images/labels)
│   └── models/              # Place .pt files here (yolo11n.pt, pitch_model.pt)
├── src/
│   ├── data_engine/         # Scripts to handle SoccerNet data
│   │   ├── __init__.py
│   │   └── soccernet_parser.py
│   ├── inference/           # Core AI Logic
│   │   ├── __init__.py
│   │   ├── detector.py      # YOLO11 Wrapper
│   │   ├── tracker.py       # ByteTrack + Ball Interpolation
│   │   ├── team.py          # Team Color Classifier (KMeans)
│   │   └── pipeline.py      # Main System Loop (The Orchestrator)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── geometry.py      # Homography & ViewTransformer
│   │   └── viz.py           # Annotators (Boxes, Radar, Text)
│   └── web/                 # Visualization
│       ├── __init__.py
│       └── main.py          # FastAPI App
├── notebooks/               # For experimentation
├── requirements.txt
├── Dockerfile
└── README.md

```

* * * * *

2\. Dependencies (`requirements.txt`)
-------------------------------------

Copy this content. It includes the exact libraries needed for the notebook reuse and the pipeline.

Plaintext

```
# Core Data & Vision
numpy>=1.24.0
opencv-python-headless>=4.8.0
tqdm
scikit-learn  # For KMeans (Team Classification)
pillow

# AI & Inference
torch>=2.0.0
ultralytics>=8.1.0  # YOLO11
supervision>=0.18.0 # Critical: Handles Tracking & Visualization
inference>=0.9.12   # Roboflow inference tools (for Pitch Model)

# Web & API
fastapi
uvicorn[standard]
python-multipart    # Required for MJPEG streaming

```

* * * * *

3\. Detailed Component Specifications
-------------------------------------

Here is the instruction set for every Python file.

### A. Data Engineering Layer (`src/data_engine/`)

**File:** `soccernet_parser.py`

-   **Purpose:** Converts SoccerNet "Tracking" JSONs into YOLO format.

-   **Key Function:** `parse_soccernet_tracking(input_dir, output_dir)`

-   **Logic:**

    1.  Iterate through the SoccerNet folder structure.

    2.  Open `Gt-Chunks.json`.

    3.  Map SoccerNet classes to ID: `0: Ball`, `1: Player`, `2: Referee`, `3: Goalkeeper`.

    4.  Extract bbox `[x, y, w, h]` (absolute pixels).

    5.  **Critical:** Convert to YOLO format `[x_center/img_w, y_center/img_h, w/img_w, h/img_h]`.

    6.  Save image frame (using `cv2`) and corresponding `.txt` label file.

### B. Inference Layer (`src/inference/`)

**File:** `detector.py`

-   **Class:** `ObjectDetector`

-   **Inputs:** `model_path` (str), `confidence` (float).

-   **Logic:**

    -   Load YOLO11 model (`ultralytics.YOLO`).

    -   Method `predict(frame)`: Returns `supervision.Detections` object.

    -   **Optimization:** Add a method `predict_pitch(frame)` that uses the Roboflow `inference` API (reusing the `PITCH_DETECTION_MODEL` from the notebook) to find field keypoints. This runs *only* on the first few frames.

**File:** `tracker.py`

-   **Class:** `bTrack`

-   **Inputs:** `frame_rate` (int).

-   **Logic:**

    -   Initialize `sv.ByteTrack`.

    -   Method `update(detections)`: Returns tracked objects.

    -   **Reuse Feature:** Implement `replace_outliers_based_on_distance` (from Notebook Cell 57) here. Before returning tracks, check the Ball ID. If it jumps > 500px, discard it.

    -   **Interpolation:** If ball is missing for < 5 frames, predict its location linearly.

**File:** `team.py`

-   **Class:** `TeamClassifier`

-   **Logic (Reused from Notebook Cells 31-35):**

    -   Uses `sklearn.cluster.KMeans(n_clusters=2)`.

    -   Method `fit(crops)`: Takes ~50 player crops from the first 5 seconds of video. Extracts average color. Trains KMeans.

    -   Method `predict(crop)`: Returns `0` (Team A) or `1` (Team B).

    -   **Optimization:** Unlike the notebook which uses SigLIP (heavy), simply use `mean_color = crop.mean(axis=(0,1))` for the feature vector. It's fast and accurate enough for distinct jerseys.

**File:** `pipeline.py` (The Brain)

-   **Class:** `SportsPipeline`

-   **Attributes:** `detector`, `tracker`, `team_classifier`, `view_transformer`.

-   **Logic (The "While True" Loop):**

    1.  **Init:** Run `detector.predict_pitch` to get 4 pitch corners. Initialize `ViewTransformer` (see Utils).

    2.  **Loop:** Get Frame.

    3.  **Detect:** Run YOLO11 -> Get Detections.

    4.  **Track:** Run ByteTrack -> Get Tracks.

    5.  **Team:** If `frame_count < 60`, collect crops. If `frame_count == 60`, train `TeamClassifier`. If `frame_count > 60`, predict team ID for every player.

    6.  **Radar:** Extract bottom-center `(x,y)` of every player. Pass to `view_transformer`.

    7.  **Viz:** Draw everything on the frame AND draw the 2D Radar on a separate blank image.

    8.  **Output:** Combine Camera Frame + Radar Image side-by-side.

### C. Utilities Layer (`src/utils/`)

**File:** `geometry.py`

-   **Class:** `ViewTransformer` (**Direct Copy from Notebook Cell 47**)

-   **Logic:**

    -   `__init__(source_points, target_points)`: Computes Homography Matrix using `cv2.findHomography`.

    -   `transform_points(points)`: Maps camera pixels to 2D pitch coordinates.

**File:** `viz.py`

-   **Logic:**

    -   Initialize `sv.BoxAnnotator`, `sv.LabelAnnotator` (from Notebook).

    -   Function `draw_radar(points, team_ids)`: Creates a green rectangle (pitch) and draws dots (players) based on transformed coordinates.

### D. Web Layer (`src/web/`)

**File:** `main.py`

-   **Framework:** FastAPI.

-   **Endpoint:** `@app.get("/video_feed")`

-   **Logic:**

    -   Instantiate `SportsPipeline`.

    -   Use a generator function `generate_frames()`.

    -   Loop: `frame = pipeline.process_next_frame()`.

    -   Encode `frame` to JPEG.

    -   Yield as `multipart/x-mixed-replace` (Standard MJPEG).

* * * * *

4\. Execution Plan (Step-by-Step for You)
-----------------------------------------

1.  **Setup:** Run `pip install -r requirements.txt`.

2.  **Data:**

    -   Download one "Tracking" folder from SoccerNet.

    -   Run `python src/data_engine/soccernet_parser.py`.

    -   *Result:* You now have a folder of images and `.txt` labels.

3.  **Training:**

    -   Run `yolo detect train model=yolo11n.pt data=your_data.yaml epochs=20`.

    -   *Result:* `runs/detect/train/weights/best.pt`. Move this to `data/models/`.

4.  **Code Logic:**

    -   Implement `src/utils/geometry.py` first (Copy/Paste from notebook).

    -   Implement `src/inference/detector.py` next.

5.  **Run:**

    -   `uvicorn src.web.main:app --reload`

    -   Open browser to `localhost:8000/video_feed`.