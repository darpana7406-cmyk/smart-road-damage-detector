# 🚧 Smart Road Damage Detector using Deep Learning

An AI-powered road damage detection and classification system that identifies **potholes and different types of road cracks** from dashboard, smartphone, or webcam images using **YOLOv8n**.

The system is designed as a **low-cost, real-time alternative to manual road inspection**, with potential applications in smart cities, road maintenance, municipal infrastructure monitoring, and intelligent transportation systems.

---

## 👩‍💻 Project Information

| Details | Information |
|---|---|
| **Project Title** | Smart Road Damage Detector using Deep Learning |
| **Author** | Darpana Shintre |
| **Roll No.** | 1022411055 |
| **Mentor** | Dr. Hrishikesh Vanjari |
| **Institution** | DES Pune University |
| **Program** | Third Year Electronics & Communication Engineering (AI-ML) |
| **Division** | A |
| **Project Type** | Final Mini Project |
| **Submission Date** | 10 October 2026 |

---

## 📌 Overview

Road damage such as potholes and cracks is one of the major problems affecting road safety, vehicle maintenance, and transportation efficiency.

Traditional road inspection methods generally depend on:

- Manual visual inspection
- Periodic surveys
- Specialized road-monitoring vehicles
- LiDAR and 3D scanning systems
- Expensive sensing infrastructure

These approaches can be **time-consuming, expensive, and difficult to scale** across large road networks.

This project proposes a **computer-vision-based road damage detection system** using **YOLOv8n**, a lightweight object-detection model capable of identifying multiple road-damage categories from ordinary camera images.

The system can process images captured using:

- 📷 Smartphone cameras
- 🚗 Vehicle dashboard cameras
- 🎥 Webcams
- 📹 Video streams

The detected damage can then be used for automated road-condition assessment and future integration with GPS-based road-health mapping systems.

---

# 🎯 Problem Statement

Manual inspection of roads is slow, expensive, and potentially dangerous for inspectors.

Road defects such as potholes and cracks can:

- Cause vehicle damage
- Increase accident risk
- Reduce driving comfort
- Create traffic disruptions
- Increase road maintenance costs
- Require frequent manual inspection

Existing automated systems may require expensive hardware such as LiDAR or specialized 3D sensors.

Therefore, there is a need for a **low-cost, camera-based system capable of detecting and classifying road damage automatically and in real time**.

---

# 🎯 Objectives

The primary objective of this project is to develop a deep-learning-based system capable of detecting and classifying road damage from camera images.

### Main objectives

1. Detect road damage using a deep-learning object detection model.
2. Classify road damage into four major categories.
3. Use low-cost RGB camera images instead of specialized hardware.
4. Achieve real-time inference on consumer GPU hardware.
5. Provide support for image and webcam-based detection.
6. Develop a foundation for GPS-based road damage mapping.
7. Estimate damage severity and size as future extensions.
8. Provide a scalable solution for smart-city road monitoring.

---

# 🛣️ Damage Categories

The model detects four types of road damage:

| Class | Description |
|---|---|
| **Pothole** | Localized depression or hole in the road surface |
| **Longitudinal Crack** | Crack running approximately parallel to the direction of the road |
| **Transverse Crack** | Crack running approximately perpendicular to the direction of the road |
| **Alligator Crack** | Interconnected network of cracks resembling alligator skin |

---

# 📊 Dataset

## RDD2022 — Road Damage Dataset 2022

The project uses the **Road Damage Dataset 2022 (RDD2022)**, introduced by Arya et al. and published in *IEEE Access*.

### Dataset Source

**Kaggle:**  
https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022

**Research / Dataset Repository:**  
https://github.com/sekilab/RoadDamageDetector

### Dataset Split Used

| Dataset | Number of Images |
|---|---:|
| Training | 18,772 |
| Validation | 3,921 |
| Testing | 3,968 |
| **Total** | **≈ 26,700** |

### Classes Used

The original road-damage dataset contains multiple road-damage categories. For this project, the dataset was mapped into four target classes:

```text
0 → pothole
1 → longitudinal_crack
2 → transverse_crack
3 → alligator_crack