# Pedestrian Behaviour Analysis For Driving

A project for analyzing pedestrian behavior using YOLOv8 for object detection, face detection, and pose estimation.

## Project Structure

```
├── gen_dataset6.py          # Script for generating dataset
├── requirements.txt         # Python dependencies
├── inference/              # Inference results and data (not tracked in git)
│   ├── images/            # Output images
│   └── videos/            # Output videos
└── Models/                 # YOLOv8 model files (download separately)
    ├── yolov8m.pt         # YOLOv8 medium model
    ├── yolov8n-face.pt    # YOLOv8 nano face detection model
    └── yolov8s-pose.pt    # YOLOv8 small pose estimation model
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

- Python 3.8+
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

Sawon Paul
