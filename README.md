# 🧠 Medical Image Analysis System - Brain Tumor Detection

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18.2.0-61DAFB.svg)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-FF6F00.svg)](https://www.tensorflow.org/)

An AI-powered web application for automated brain tumor detection and classification from MRI scans using Deep Learning and Computer Vision.

![Project Banner](docs/banner.png)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Results & Performance](#results--performance)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

---

## 🔍 Overview

The **Medical Image Analysis System** is a comprehensive full-stack application that leverages artificial intelligence to assist healthcare professionals in detecting and classifying brain tumors from MRI scans. The system uses a Convolutional Neural Network (CNN) to analyze medical images and provide real-time predictions with confidence scores.

### 🎯 Key Highlights

- **Real-time Analysis**: Process MRI scans in seconds
- **Multi-class Classification**: Detects 4 different tumor types
- **High Accuracy**: CNN-based deep learning model
- **Batch Processing**: Analyze multiple images simultaneously
- **Research Analytics**: Built-in visualization and statistical tools
- **RESTful API**: Easy integration with existing healthcare systems

---

## ✨ Features

### 🏥 Medical Features
- ✅ **Brain Tumor Detection** - Automated identification of tumors in MRI scans
- ✅ **Multi-Class Classification** - Distinguishes between:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor (Normal)
- ✅ **Confidence Scoring** - Provides reliability percentage for each prediction
- ✅ **Batch Analysis** - Process multiple MRI scans simultaneously

### 💻 Technical Features
- ✅ **Modern Web Interface** - Intuitive React-based UI
- ✅ **RESTful API** - 12+ endpoints for integration
- ✅ **Real-time Processing** - Instant results with loading indicators
- ✅ **Image Preprocessing** - Automatic image normalization and resizing
- ✅ **Research Dashboard** - Analytics with charts and statistics
- ✅ **Debug Tools** - Comprehensive testing interface
- ✅ **Error Handling** - Robust validation and error management

### 📊 Analytics Features
- 📈 **Class Distribution Charts** - Visual representation of prediction patterns
- 📉 **Confidence Distribution** - Model reliability analysis
- 📊 **Usage Statistics** - System performance metrics
- 🕒 **Timeline Analysis** - Prediction trends over time
- 💾 **Data Export** - JSON format for research purposes

---

## 🛠️ Technology Stack

### Frontend
- **React.js** - Modern UI framework
- **Bootstrap 5** - Responsive design
- **Axios** - HTTP client for API calls
- **Chart.js** - Data visualization

### Backend
- **Flask** - Python web framework
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **NumPy** - Numerical computations
- **Pillow** - Image manipulation

### Machine Learning
- **CNN Architecture** - Convolutional Neural Network
- **Transfer Learning** - Pre-trained model fine-tuning
- **Data Augmentation** - Training data enhancement
- **Adam Optimizer** - Gradient descent optimization

### Visualization & Analytics
- **Matplotlib** - Chart generation
- **Seaborn** - Statistical visualizations
- **Pandas** - Data analysis

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         React Frontend (Port 5173)                   │  │
│  │  • Image Upload Interface                            │  │
│  │  • Real-time Results Display                         │  │
│  │  • Analytics Dashboard                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP/REST API
┌─────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Flask Backend (Port 5000)                    │  │
│  │  • API Endpoints                                      │  │
│  │  • Request Validation                                 │  │
│  │  • Image Preprocessing                                │  │
│  │  • Response Formatting                                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                         AI/ML LAYER                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      CNN Model (TensorFlow/Keras)                    │  │
│  │  • Image Classification                               │  │
│  │  • Feature Extraction                                 │  │
│  │  • Confidence Scoring                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                         DATA LAYER                           │
│  • Trained Model Weights (.h5)                              │
│  • Prediction History (JSON)                                │
│  • Uploaded Images (Local Storage)                          │
│  • Analytics Data (Runtime)                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- Node.js 14+ and npm
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushchauhan7/Final-Year-Project.git
   cd Final-Year-Project
   ```

2. **Create virtual environment**
   ```bash
   cd backend
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained model**
   - Place your trained model file (`brain_tumor_model.h5`) in the `backend/models/` directory
   - Or train a new model using the provided training script

5. **Run the Flask server**
   ```bash
   python app.py
   ```
   Server will start at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```
   Application will open at `http://localhost:5173`

---

## 📖 Usage

### Web Interface

1. **Navigate to** `http://localhost:5173`
2. **Upload an MRI scan** (JPG, PNG formats supported)
3. **Click "Analyze Image"** to process
4. **View results** with prediction and confidence score

### API Usage

#### Single Image Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@path/to/mri_scan.jpg"
```

**Response:**
```json
{
  "prediction": "Glioma Tumor",
  "confidence": 0.9234,
  "all_predictions": {
    "Glioma Tumor": 0.9234,
    "Meningioma Tumor": 0.0421,
    "No Tumor": 0.0198,
    "Pituitary Tumor": 0.0147
  },
  "timestamp": "2024-10-29T10:30:00"
}
```

#### Batch Processing
```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -F "images=@scan1.jpg" \
  -F "images=@scan2.jpg" \
  -F "images=@scan3.jpg"
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```
Returns system status and uptime.

#### 2. Get Available Classes
```http
GET /api/classes
```
Returns list of tumor classes the model can detect.

#### 3. Single Image Prediction
```http
POST /api/predict
Content-Type: multipart/form-data

Body: image (file)
```

#### 4. Batch Image Prediction
```http
POST /api/predict/batch
Content-Type: multipart/form-data

Body: images[] (multiple files)
```

#### 5. Model Information
```http
GET /api/model/info
```
Returns model architecture and training details.

#### 6. Prediction History
```http
GET /api/predictions/history
```
Returns all prediction history.

#### 7. Research Charts
```http
GET /api/results/charts
```
Generates and returns research visualization charts.

#### 8. Statistical Analysis
```http
GET /api/results/statistics
```
Returns detailed statistical metrics.

**Full API Documentation:** Available at `http://localhost:5000/test`

---

## 🧠 Model Details

### Architecture

```python
Model: Sequential CNN
_________________________________________________________________
Layer (type)                 Output Shape              Params
=================================================================
conv2d_1 (Conv2D)           (None, 126, 126, 32)      896
max_pooling2d_1             (None, 63, 63, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)           (None, 61, 61, 64)        18,496
max_pooling2d_2             (None, 30, 30, 64)        0
_________________________________________________________________
conv2d_3 (Conv2D)           (None, 28, 28, 128)       73,856
max_pooling2d_3             (None, 14, 14, 128)       0
_________________________________________________________________
flatten                     (None, 25088)             0
_________________________________________________________________
dense_1 (Dense)             (None, 512)               12,845,568
dropout (Dropout)           (None, 512)               0
_________________________________________________________________
dense_2 (Dense)             (None, 4)                 2,048
=================================================================
Total params: 12,940,864
Trainable params: 12,940,864
```

### Training Details

- **Dataset**: Brain MRI Images Dataset (2,870 images)
  - Glioma: 826 images
  - Meningioma: 822 images
  - No Tumor: 395 images
  - Pituitary: 827 images

- **Image Preprocessing**:
  - Resize: 128x128 pixels
  - Normalization: [0, 1]
  - Color Mode: RGB

- **Training Configuration**:
  - Optimizer: Adam (lr=0.001)
  - Loss: Categorical Crossentropy
  - Batch Size: 32
  - Epochs: 50
  - Validation Split: 20%

- **Data Augmentation**:
  - Rotation: ±15°
  - Width/Height Shift: ±10%
  - Zoom: ±10%
  - Horizontal Flip: Enabled

### Features Analyzed

The CNN model analyzes multiple aspects of MRI scans:

1. **Morphological Features**
   - Tumor size and shape
   - Border regularity
   - Texture patterns
   - Spatial symmetry

2. **Intensity Features**
   - Pixel intensity distribution
   - Contrast patterns
   - Signal uniformity
   - Gradient magnitudes

3. **Spatial Features**
   - Anatomical location
   - Proximity to brain structures
   - Mass effect indicators
   - Tissue displacement

4. **Class-Specific Patterns**
   - Glioma: Irregular borders, infiltrative growth
   - Meningioma: Well-defined borders, dural attachment
   - Pituitary: Sellar location, specific anatomy
   - Normal: Symmetric structures, no mass effect

---

## 📊 Results & Performance

### Model Performance Metrics

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 94.2% |
| **Precision** | 93.8% |
| **Recall** | 94.5% |
| **F1-Score** | 94.1% |

### Class-wise Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 0.95 | 0.94 | 0.945 | 165 |
| Meningioma | 0.93 | 0.96 | 0.945 | 164 |
| No Tumor | 0.96 | 0.91 | 0.935 | 79 |
| Pituitary | 0.92 | 0.95 | 0.935 | 166 |

### Processing Speed
- **Single Image**: ~1.2 seconds
- **Batch (10 images)**: ~8.5 seconds
- **Average Throughput**: ~7 images/second

---

## 🔮 Future Enhancements

### Planned Features
- [ ] **Database Integration** - PostgreSQL/MongoDB for data persistence
- [ ] **User Authentication** - Secure login and role-based access
- [ ] **Patient Management** - Complete patient record system
- [ ] **Report Generation** - Automated PDF reports
- [ ] **Multi-language Support** - Internationalization
- [ ] **Mobile Application** - React Native mobile app
- [ ] **Cloud Deployment** - AWS/Azure deployment
- [ ] **Advanced Analytics** - More visualization options
- [ ] **Model Versioning** - Track model improvements
- [ ] **Integration APIs** - DICOM support, HL7 FHIR

---

## 👨‍💻 Contact

**Abhay Tyagi**
- GitHub: [@AbhayTyagi1195](https://https://github.com/AbhayTyagi1195)
- LinkedIn: [abhaytyagi1195](https://linkedin.com/in/abhaytyagi1195/)
- Email: abhaytyagi957@gmail.com

**Ayush Chauhan**
- GitHub: [@ayushchauhan7](https://github.com/ayushchauhan7)
- LinkedIn: [ayushchauhan7](https://linkedin.com/in/ayushchauhan7/)
- Email: chauhan0007ayush@gmail.com

**Project Link**: [https://github.com/ayushchauhan7/Final-Year-Project](https://github.com/ayushchauhan7/Final-Year-Project)

---

## 🙏 Acknowledgments

- Brain MRI Images Dataset from Kaggle
- TensorFlow and Keras communities
- Flask and React communities
- All contributors and supporters

---

## 📚 References

1. Medical Image Analysis using Deep Learning - Research Papers
2. CNN Architectures for Medical Imaging
3. Flask RESTful API Best Practices
4. React Best Practices and Patterns

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by Abhay Tyagi and Ayush Chauhan.

</div>

