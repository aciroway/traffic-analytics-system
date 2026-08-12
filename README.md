
```markdown
# 🚗 Traffic Analytics System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-000000.svg?style=flat)](https://github.com/ultralytics/ultralytics)

Real-time vehicle detection, multi-object tracking, directional line crossing analytics, speed estimation, and traffic violation detection pipeline built with **YOLO**, **ByteTrack**, **Supervision**, and **OpenCV**.

---

## 📌 Overview & Architecture

This repository contains an end-to-end Computer Vision pipeline designed for intelligent transportation systems (ITS) and Smart City video analytics.


```

Video Input (File / RTSP)
│
▼
┌───────────────┐
│ YOLO Inference│ ──► Object Detection (Cars, Trucks, Buses, Motorcycles)
└───────┬───────┘
▼
┌───────────────┐
│   ByteTrack   │ ──► Multi-Object Tracking & Class Stabilization
└───────┬───────┘
▼
┌───────────────┐
│  Supervision  │ ──► Line Crossing Analytics (IN / OUT Counters)
└───────┬───────┘
▼
Annotated Output (Video + Analytics Events Log)

```

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Vehicle Detection** | High-precision object detection leveraging YOLO architectures. |
| **Object Tracking** | Multi-object tracking powered by ByteTrack with ID persistence. |
| **Traffic Counting** | Bidirectional line-crossing counter (`IN` / `OUT`) using Supervision. |
| **Hardware Acceleration** | Automatic CUDA GPU detection with fallback to CPU. |
| **Modular Design** | Decoupled pipeline components for easy extension (speed, violations, logging). |

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, Ultralytics YOLO
* **Computer Vision:** OpenCV, Supervision
* **Object Tracking:** ByteTrack

---

## 🚀 Quick Start

### 1. Clone & Environment Setup

```bash
git clone [https://github.com/aciroway/traffic-analytics-system.git](https://github.com/aciroway/traffic-analytics-system.git)
cd traffic-analytics-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Execution

Run the main pipeline:

```bash
python main.py

```

---

## 📂 Project Structure

```text
traffic-analytics-system/
├── .gitignore          # Excluded files (videos, weights, venv)
├── main.py             # Main video processing pipeline
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

```

---

## 📜 License

Distributed under the MIT License.


