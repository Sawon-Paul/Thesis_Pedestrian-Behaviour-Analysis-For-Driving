# 🚶 Pedestrian Behaviour Analysis For Driving (AEB Analytics Pipeline)

A high-performance, GPU-adaptive pipeline for analyzing pedestrian behavior, intent, and collision risk. This project leverages an ensemble of YOLOv8 models, MediaPipe, and HSEmotion to generate rich, frame-by-frame dataset annotations for Autonomous Emergency Braking (AEB) and social-context models.

## 🌟 Key Features

- **🚀 GPU-Adaptive Extreme Performance:** Auto-tunes inference resolution, thread pools, and VRAM limits based on hardware (scales from CPU fallback up to RTX 4090 / H100 datacenters). Utilizes `torch.compile`, CUDA streams, and asynchronous pinned memory.
- **🧠 Unified Multi-Model Pipeline:** 
  - **YOLOv8s-pose:** Pedestrian tracking, bounding boxes, and 17-point COCO keypoints.
  - **YOLOv8m:** High-accuracy smartphone and held-item detection.
  - **YOLOv8n-face + HSEmotion:** Real-time facial emotion recognition.
  - **MediaPipe Hands:** Fallback for precise STOP/yield hand gestures.
- **🚗 AEB Pedestrian Logic:** Predictive collision pathing, crossing intent recognition, monocular pinhole depth estimation, and social-context tracking (distance to peers).
- **🗺️ GPU-Accelerated Bird's Eye View (BEV):** Real-time, top-down distance map with trajectory trails and ego-vehicle proximity rings.
- **💾 Zero-Bottleneck Data Continuity:** Non-destructive checkpointing (`checkpoint.json`). Pausing, resuming, and restarting effortlessly across multiple video sessions without data truncation. Asynchronous IO prevents frame freezing.

## Project Structure

```
├── gen_dataset6.py           # Core analytics and dataset generation script
├── requirements.txt          # Python dependencies
├── checkpoint.json           # Auto-generated DB tracking paused/resumed video states
├── dataset_master.csv        # Master dataset (appended continuously)
├── dataset.json              # Master JSON schema (appended continuously)
├── inference/                # Generated assets (ignored in version control)
│   ├── videos/               # Place input videos here
│   └── images/               # Annotated snapshots triggered by dataset events
└── Models/                   # Model weights directory
    ├── yolov8s-pose.pt       
    ├── yolov8m.pt            
    └── yolov8n-face.pt       
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Sawon-Paul/Thesis_Pedestrian-Behaviour-Analysis-For-Driving.git
cd Thesis_Pedestrian-Behaviour-Analysis-For-Driving
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 Models

The model files (.pt) are not included in the repository due to size limitations. Download them from Ultralytics:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt'); YOLO('yolov8n-face.pt'); YOLO('yolov8s-pose.pt')"
```

Or download them manually and place in the project root directory.

### 5. Run Dataset Generation
```bash
python gen_dataset6.py
```

## Requirements

- Python 3.10.0
- PyTorch
- YOLOv8
- OpenCV
- (See requirements.txt for complete list)

## Notes

- The `inference/` directory contains large output files and is not tracked in version control
- Model files (.pt) are not tracked due to GitHub file size limitations
- Ensure you have sufficient disk space for inference results

## License

[Add your license information here]

## Author

Sawon Paul, Prottoy Saha, Rifat Farzad Azad, Sanjida Islam Susmita, Tanjina Tansnim
