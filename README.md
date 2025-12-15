# 🚦 AI-Enabled Traffic Management System

An intelligent real-time traffic violation detection system using deep learning and computer vision to automatically identify traffic rule violations and alert authorities.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![YOLOv3](https://img.shields.io/badge/YOLO-v3-red.svg)](https://pjreddie.com/darknet/yolo/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

This project addresses the growing challenge of traffic violations in urban areas through automated detection using artificial intelligence. The system achieves **88.3% accuracy** in identifying various traffic violations in real-time, eliminating the need for manual monitoring by traffic police.

### Problem Statement
- Manual traffic violation detection is slow and inefficient
- Traffic police cannot capture multiple violations simultaneously
- Human intervention leads to delays and reduced accuracy

### Solution
An end-to-end automated system that:
- Detects violations in real-time using computer vision
- Identifies vehicle license plates using OCR
- Automatically alerts authorities via SMS with violation details

## ✨ Features

### 🪖 Helmet Detection
- Real-time detection of riders with and without helmets on two-wheelers
- Uses YOLOv3 with custom-trained weights for helmet classification
- Automatically captures violator images for evidence

### 👥 Triple Riding Detection
- Identifies multiple riders (>2) on two-wheelers using person detection
- Employs Deconvolutional Neural Network-based YOLO algorithm
- Classifies vehicles as rule-breach or compliant

### 🚗 License Plate Recognition
- Automatic number plate detection and extraction using YOLO
- Optical Character Recognition (OCR) using TensorFlow Lite
- Perspective transformation for accurate text extraction

### 📱 SMS Alert System
- Integration with Twilio API for instant SMS notifications
- Sends violation details with vehicle registration number to authorities
- Automated challan generation and notification to vehicle owners

### 🖥️ User-Friendly GUI
- Tkinter-based graphical interface for traffic authorities
- Single dashboard to monitor multiple violation types
- Real-time video processing and violation logging

## 🏗️ System Architecture

```
┌─────────────────┐
│   Video Input   │
│  (Live/File)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Frame Extract  │
│    (OpenCV)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLO Detection │
│   (YOLOv3 CNN)  │
└────────┬────────┘
         │
         ├──────────────┬────────────────┐
         ▼              ▼                ▼
   ┌──────────┐  ┌──────────┐    ┌──────────┐
   │ Helmet   │  │  Triple  │    │  Plate   │
   │Detection │  │  Riding  │    │Detection │
   └─────┬────┘  └─────┬────┘    └─────┬────┘
         │             │               │
         └─────────────┴───────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  OCR Processing │
              │ (TensorFlow Lite)│
              └────────┬─────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  SMS Alert      │
              │  (Twilio API)   │
              └─────────────────┘
```

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.7+** - Primary programming language
- **OpenCV** - Image processing and video frame extraction
- **YOLOv3** - Object detection (vehicles, helmets, persons)
- **TensorFlow Lite** - OCR for license plate text recognition
- **NumPy** - Numerical computations and array operations

### Frameworks & Libraries
- **Tkinter** - GUI development for traffic management interface
- **Twilio API** - SMS notification system
- **Pandas** - Data logging and management
- **Darknet** - Deep learning framework for YOLO

### Machine Learning Models
- **YOLOv3** - Convolutional Neural Network for object detection
- **Random Forest** - Classification for violation categorization
- **OCR Model** - Custom-trained text recognition model

### Infrastructure
- **PyCharm** - Development IDE
- **Windows 8/10** - Operating system

## 📦 Installation

### Prerequisites
```bash
# Check Python version
python --version  # Should be 3.7 or higher
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/modhisathvik7733/Artificial-intelligence-enabled-traffic-management-system.git
cd Artificial-intelligence-enabled-traffic-management-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-python==4.5.5.64
numpy==1.21.6
tensorflow==2.8.0
twilio==7.16.0
pandas==1.3.5
Pillow==9.0.1
```

### Step 3: Download YOLO Weights
```bash
# Download YOLOv3 weights for general object detection
wget https://pjreddie.com/media/files/yolov3.weights

# Download custom-trained helmet detection weights
# Place in project root directory
```

### Step 4: Configure Twilio API (Optional)
1. Create a free account at [Twilio](https://www.twilio.com/)
2. Get your Account SID and Auth Token
3. Update credentials in the SMS notification module:
```python
# In your SMS module
account_sid = "YOUR_ACCOUNT_SID"
auth_token = "YOUR_AUTH_TOKEN"
from_number = "YOUR_TWILIO_NUMBER"
to_number = "AUTHORITY_PHONE_NUMBER"
```

### Step 5: Configure OpenALPR API (for OCR)
```python
SECRET_KEY = "YOUR_OPENALPR_SECRET_KEY"
```

## 🚀 Usage

### Running the GUI Application
```bash
python main.py
```

This launches the traffic management interface with options for:
- Helmet Detection (Image/Video)
- Triple Riding Detection (Image/Video)
- License Plate Detection
- Live Traffic Monitoring

### Running Individual Modules

**Helmet Detection on Video:**
```bash
python HelmetdetectionYOLOV3.py
```

**Triple Riding Detection:**
```bash
python yolodetectionwebcam1.py
```

**License Plate Recognition:**
```bash
python live1.py  # For live detection
```

**OCR Text Extraction:**
```bash
python ocr_detection.py
```

### Input Formats
- **Video Files:** `.mp4`, `.avi`
- **Image Files:** `.jpg`, `.png`
- **Live Feed:** Webcam or IP camera stream

### Output
- Detected violations saved in `output/` folder
- Violation logs stored in `data.csv`
- SMS alerts sent automatically to configured numbers

## 📊 Results

### Performance Metrics
| Metric | Value |
|--------|-------|
| Overall Accuracy | 88.3% |
| Helmet Detection Accuracy | 85-90% |
| Triple Riding Detection | 82-87% |
| License Plate Recognition | 75-80% |
| Real-time Processing | ✅ Yes |

### Detection Speed
- **Video Processing:** ~15-20 FPS on standard CPU
- **Image Detection:** ~0.5-1 second per image
- **OCR Processing:** ~2-3 seconds per plate

### Advantages Over Existing Systems
✅ Automated 24/7 monitoring  
✅ Faster than manual detection  
✅ Can detect multiple violations simultaneously  
✅ Cost-effective compared to human resources  
✅ Higher accuracy than traditional methods  

## 📁 Project Structure

```
Artificial-intelligence-enabled-traffic-management-system/
│
├── cfg/
│   ├── yolov3.cfg              # YOLOv3 configuration
│   └── yolov3-obj.cfg          # Custom helmet detection config
│
├── weights/
│   ├── yolov3.weights          # Pre-trained YOLO weights
│   └── yolov3-obj2400.weights  # Custom helmet weights
│
├── data/
│   ├── coco.names              # COCO dataset labels
│   └── obj.names               # Custom object labels
│
├── output/
│   ├── helmets/                # Helmet violation images
│   ├── tripleride/             # Triple riding violations
│   └── numberplates/           # Extracted license plates
│
├── HelmetdetectionYOLOV3.py    # Helmet detection module
├── yolodetectionwebcam1.py     # Triple riding detection
├── live1.py                    # License plate detection
├── ocr_detection.py            # OCR text extraction
├── main.py                     # GUI application entry point
├── data.csv                    # Violation logs
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🔮 Future Enhancements

- [ ] Add more violation types (wrong-way driving, signal violations)
- [ ] Implement cloud-based processing for scalability
- [ ] Integrate with traffic management databases
- [ ] Add mobile application for authorities
- [ ] Implement real-time dashboard with analytics
- [ ] Support for multiple camera feeds simultaneously
- [ ] Integration with automatic challan payment systems
- [ ] Improve OCR accuracy with advanced deep learning models
- [ ] Add night-time detection capabilities
- [ ] Implement vehicle type classification

## 👥 Contributors

**Sathvik Modhi** - [@modhisathvik7733](https://github.com/modhisathvik7733)  
**V. Guru Pavani**  
**V. Surya Sagar**

**Guided by:** Dr. T. S. Mastan Rao, Associate Professor  
**Institution:** CMR Technical Campus, JNTU Hyderabad

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv3 by Joseph Redmon and Ali Farhadi
- OpenCV community for computer vision libraries
- OpenALPR for OCR capabilities
- Twilio for SMS API integration
- CMR Technical Campus for project support


---

⭐ If you find this project useful, please consider giving it a star!
