AI Model
YOLOv8n
The project uses YOLOv8n (YOLOv8 Nano) from Ultralytics.
YOLOv8n was selected because it provides a good balance between:
- Detection accuracy
- Computational requirements
- Inference speed
- Model size
- Real-time performance
Model Specifications
Parameter	Value
Model	YOLOv8n
Parameters	≈ 2.68 Million
GFLOPs	≈ 6.8
Framework	PyTorch
GPU	NVIDIA RTX 2050
GPU Memory	4 GB
CUDA	12.1
Computer Vision	OpenCV


System Architecture
The overall workflow of the proposed system is:
                 ┌─────────────────────┐
                 │   Camera / Image    │
                 │  Smartphone/Webcam  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Image Preprocessing │
                 │      OpenCV         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     YOLOv8n         │
                 │ Object Detection    │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Road Damage Classification│
              └────────────┬─────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       Pothole        Road Cracks     Damage Box
                                       Detection
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                 ┌─────────────────────┐
                 │ Detection Results   │
                 │ + Confidence Score  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Future Extensions  │
                 │ GPS / Severity /    │
                 │ Road Health Mapping │
                 └─────────────────────┘

Results
The model was evaluated on the 3,921-image validation set.
Overall Performance
Metric	Value
mAP@50	0.548
mAP@50-95	0.287
Precision	0.611
Recall	0.516
Inference Time	9.7 ms/image
Approx. GPU FPS	≈103 FPS

The results demonstrate that the lightweight YOLOv8n model can perform road-damage detection at real-time speeds on a consumer-grade NVIDIA RTX 2050 4 GB GPU.

Per-Class Performance
Class	Instances	Precision	Recall	mAP@50
Alligator Crack	3,116	0.663	0.671	0.705
Longitudinal Crack	3,890	0.595	0.511	0.536
Transverse Crack	1,769	0.584	0.527	0.531
Pothole	965	0.603	0.353	0.421

Results Analysis
Best Performing Class — Alligator Crack
The alligator crack class achieved the highest mAP@50:
mAP@50 = 0.705
This can be attributed to its distinctive interconnected texture and visual appearance, which makes it comparatively easier for the object detector to recognize.
Weakest Performing Class — Pothole
Pothole detection produced the lowest mAP@50:
mAP@50 = 0.421
The relatively lower recall indicates that a considerable number of potholes were missed.
One major factor is the class imbalance in the dataset, with substantially fewer pothole examples compared with some crack categories.

Real-Time Performance
The model achieved approximately:
9.7 ms/image
≈ 103 FPS on an NVIDIA RTX 2050 4 GB GPU.
This demonstrates the potential of YOLOv8n for real-time road monitoring applications using relatively affordable hardware.

Project Structure
PBEL/
│
├── src/
│   ├── train.py
│   ├── validate.py
│   ├── detect.py
│   ├── gps.py
│   ├── severity.py
│   ├── measure.py
│   ├── report.py
│   └── remap.py
│
├── data/
│   └── yolo/
│       ├── data.yaml
│       │
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       │
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       │
│       └── test/
│           ├── images/
│           └── labels/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

Module Description
File	Purpose
train.py	Trains the YOLOv8n model
validate.py	Evaluates model performance and calculates metrics
detect.py	Performs image and webcam inference
gps.py	GPS tagging functionality
severity.py	Damage severity estimation
measure.py	Damage size measurement
report.py	Automated report generation
remap.py	Dataset class remapping utility
app.py	Flask web application prototype
data.yaml	YOLO dataset configuration


Note: The gps.py, severity.py, measure.py, and report.py modules are included as extensions/modules for the broader road-monitoring system. Their functionality can be further developed and integrated into the final deployment pipeline.

Installation
1. Clone the Repository
git clone YOUR_REPO_URL
cd PBEL

Replace YOUR_REPO_URL with the URL of your GitHub repository.
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

Usage
1. Train the Model
To train YOLOv8n on the configured dataset:
python src/train.py

The training configuration is defined in the training script and data.yaml.
2. Validate the Model
Run validation and generate performance metrics:
python src/validate.py

This evaluates the model using metrics such as:
- Precision
- Recall
- mAP@50
- mAP@50-95
3. Detect Damage in an Image
Run inference on a single image:
python src/detect.py path\to\image.jpg

Example:
python src/detect.py test_images/road.jpg

The output displays detected road-damage classes along with their bounding boxes and confidence scores.
4. Real-Time Webcam Detection
To perform real-time detection using a webcam:
python src/detect.py 0

Here:
0 → Default webcam

A different camera index can be used if multiple cameras are connected.
Flask Web Application
The project also includes a Flask-based web application prototype.
Start the application using:
python app.py

Then open:
http://127.0.0.1:5000

The web interface can be extended to allow users to:
- Upload road images
- Run damage detection
- Display bounding boxes
- Show detected damage categories
- Display confidence scores
- Generate road-damage reports
Hardware Used
The system was developed and tested using:
GPU
NVIDIA GeForce RTX 2050
4 GB VRAM

Software Environment
Python
PyTorch
CUDA
YOLOv8
OpenCV
Flask

The use of YOLOv8n allows the system to operate without requiring high-end GPU hardware.
Requirements
The major software dependencies include:
Python
PyTorch
Ultralytics
OpenCV
NumPy
Flask
Pandas
Matplotlib

The exact package versions are specified in:
requirements.txt

Git & File Management
Large generated files and local environments should not be committed to the repository.
The following are excluded through .gitignore:
runs/
uploads/
outputs/
venv/
*.pt
*.pth
__pycache__/

This keeps the GitHub repository lightweight and prevents trained model weights and generated outputs from unnecessarily increasing repository size.
Future Scope
The current system provides a foundation for a more comprehensive intelligent road-monitoring platform.
1. Improved Model Accuracy
Experiment with larger YOLO models such as:
YOLOv8s
YOLOv8m

to potentially improve detection accuracy.
The trade-off between model accuracy and inference speed can be studied for deployment.
2. Class Imbalance Handling
The pothole class has comparatively fewer training examples.
Future work can investigate:
- Oversampling
- Data augmentation
- Synthetic data generation
- Class-balanced training
- Focal loss
- Hard-example mining
to improve pothole recall.
3. Night-Time Detection
Performance can be improved for difficult environmental conditions such as:
- Night-time roads
- Low illumination
- Rain
- Fog
- Shadows
- Glare
- Wet road surfaces
Additional training data and image-enhancement techniques can be incorporated.
4. Mobile Deployment
The trained model can potentially be optimized for mobile devices using:
TensorFlow Lite
ONNX
NCNN

This could allow road-damage detection directly from an Android or iOS smartphone.
5. GPS-Based Road Mapping
GPS coordinates can be associated with each detected road defect.
This can enable the development of a road-health map such as:
Road Location
      ↓
GPS Coordinates
      ↓
Damage Detection
      ↓
Damage Type
      ↓
Severity
      ↓
Road Health Map

Municipal authorities could use such a system to identify locations requiring maintenance.
6. Damage Severity Estimation
Future versions can estimate the severity of detected damage based on:
- Bounding-box dimensions
- Damage area
- Crack length
- Crack density
- Pothole size
- Image perspective
The system could classify damage into:
Low
Medium
High
Critical

7. Damage Size Measurement
Computer-vision-based calibration techniques can be incorporated to estimate the physical dimensions of road damage.
For example:
Detected Pothole
      ↓
Pixel Measurement
      ↓
Camera Calibration
      ↓
Perspective Correction
      ↓
Estimated Physical Size

8. Multi-Modal Sensing
A future version can combine computer vision with other sensors such as:
- LiDAR
- Accelerometer
- Gyroscope
- Vehicle-mounted sensors
- Depth cameras
This could improve pothole-depth estimation and overall road-condition assessment.
9. Smart City Dashboard
The complete system can eventually be extended into a cloud-based dashboard for municipal corporations.
Possible features include:
- Interactive road-health maps
- GPS-tagged damage locations
- Damage severity statistics
- Historical road-condition data
- Maintenance priority ranking
- Automated reports
- Road-condition trends

Potential Applications
The proposed system can be used in:
- Smart city infrastructure monitoring
- Intelligent transportation systems
- Road maintenance planning
- Municipal corporation surveys
- Fleet management
- Smartphone-based road inspection
- Highway monitoring
- Road-health mapping
- Infrastructure analytics

References
Dataset
Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., and Sekimoto, Y.
RDD2022 — Road Damage Dataset 2022
Dataset available through:
https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022
Research repository:
https://github.com/sekilab/RoadDamageDetector
YOLOv8
Ultralytics.
YOLOv8 — Real-Time Object Detection
https://github.com/ultralytics/ultralytics

Acknowledgements
We would like to acknowledge:
- Arya et al. for the RDD2022 road-damage dataset.
- Ultralytics for the YOLOv8 object-detection framework.
- PyTorch for the deep-learning framework.
- OpenCV for computer-vision and image-processing functionality.

Author
Darpana Shintre
