# Medical Image Analysis System

A comprehensive AI-powered medical image analysis system for brain tumor detection using deep learning and modern web technologies.

## 🧠 Overview

This system provides an end-to-end solution for brain tumor detection using artificial intelligence. It combines a robust Flask backend with a modern React frontend to deliver accurate, fast, and user-friendly medical image analysis.

## ✨ Features

### Backend Features
- 🤖 **AI-Powered Analysis**: CNN-based brain tumor classification
- 🔌 **RESTful API**: Comprehensive API with multiple endpoints
- 🗄️ **Database Integration**: SQLite database for prediction history
- 🛡️ **Security**: Rate limiting, input validation, and error handling
- 📊 **Analytics**: Detailed prediction statistics and insights
- 🔧 **Debug Tools**: Comprehensive debugging and testing endpoints

### Frontend Features
- 🎨 **Modern UI**: Beautiful, responsive React interface
- 📱 **Mobile-Friendly**: Works seamlessly on all devices
- 🖱️ **Drag & Drop**: Intuitive image upload with drag-and-drop
- 📈 **Real-time Results**: Live analysis with progress indicators
- ⚠️ **Smart Warnings**: Uncertainty detection and medical disclaimers
- 🎯 **Detailed Analysis**: Comprehensive results with confidence scores

## 🏗️ Architecture

```
Medical Image Analysis System
├── backend/                 # Flask API Server
│   ├── app.py              # Main Flask application
│   ├── models/             # AI model definitions
│   ├── utils/              # Utility functions
│   ├── database.py         # Database operations
│   └── requirements.txt    # Python dependencies
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── Header.js   # Application header
│   │   │   ├── Footer.js   # Application footer
│   │   │   ├── ImageUpload.js    # Image upload component
│   │   │   └── AnalysisResults.js # Results display
│   │   ├── App.js         # Main application
│   │   └── App.css        # Global styles
│   └── package.json       # Node.js dependencies
├── notebooks/             # Jupyter notebooks for training
└── trained_models/        # AI model files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with pip
- Node.js 14+ with npm
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Medical-Image-Analysis-System
   ```

2. **Set up the backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd ../frontend
   npm install
   ```

4. **Start the system**
   
   **Option A: Use the development script**
   ```bash
   # Windows
   start-dev.bat
   
   # Linux/Mac
   chmod +x start-dev.sh
   ./start-dev.sh
   ```
   
   **Option B: Start manually**
   ```bash
   # Terminal 1 - Backend
   cd backend
   python app.py
   
   # Terminal 2 - Frontend
   cd frontend
   npm start
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000
   - Test Page: http://localhost:5000/test

## 📡 API Documentation

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Analyze brain scan image |
| `/api/health` | GET | Health check and system status |
| `/api/classes` | GET | Available tumor classes |
| `/api/model/info` | GET | Model information and configuration |

### Debug Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/debug/prediction` | POST | Detailed prediction analysis |
| `/api/debug/class-order` | POST | Test different class interpretations |
| `/api/analytics/summary` | GET | Prediction statistics |
| `/api/predictions/history` | GET | Recent prediction history |

### Example API Usage

```bash
# Health check
curl http://localhost:5000/api/health

# Analyze image
curl -X POST -F "image=@brain_scan.jpg" http://localhost:5000/api/predict
```

## 🧪 Model Information

The system uses a trained Convolutional Neural Network (CNN) for brain tumor classification:

### Supported Classes
- **Meningioma Tumor**: Tumors arising from the meninges
- **No Tumor**: Normal brain scan with no tumor detected

### Model Specifications
- **Input Size**: 224x224x3 (RGB images)
- **Architecture**: CNN with 4 Conv2D layers
- **Training**: 15 epochs with data augmentation
- **Confidence Threshold**: 30% minimum for predictions

### Performance
- **Accuracy**: Optimized for medical image analysis
- **Speed**: ~100-500ms processing time per image
- **Reliability**: High/Medium/Low confidence levels

## 🎯 Usage Guide

### For Users
1. **Upload Image**: Drag and drop or select a brain scan image
2. **Analyze**: Click "Analyze Image" to process with AI
3. **Review Results**: Check detailed analysis including:
   - Tumor detection status
   - Confidence percentage
   - Probability breakdown
   - Reliability assessment
   - Medical recommendations

### For Developers
1. **Backend Development**: Modify `backend/app.py` and related files
2. **Frontend Development**: Update components in `frontend/src/components/`
3. **Model Training**: Use notebooks in the `notebooks/` directory
4. **Testing**: Use debug endpoints for comprehensive testing

## 🔧 Development

### Backend Development
```bash
cd backend
python app.py  # Start development server
```

### Frontend Development
```bash
cd frontend
npm start      # Start React development server
```

### Testing
- **Backend**: Use the test page at `http://localhost:5000/test`
- **Frontend**: Use browser developer tools
- **Integration**: Test full workflow from upload to results

## 📊 Analytics & Monitoring

The system includes comprehensive analytics:
- Prediction history and statistics
- Confidence level distributions
- Processing time metrics
- Error rate monitoring
- User interaction analytics

## 🛡️ Security & Privacy

- **Data Privacy**: Images are processed locally and not stored permanently
- **Rate Limiting**: API includes rate limiting to prevent abuse
- **Input Validation**: Comprehensive validation for all inputs
- **Error Handling**: Secure error messages without sensitive information
- **Medical Disclaimer**: Clear warnings about educational use only

## 🚀 Deployment

### Production Backend
1. Set up production WSGI server (Gunicorn)
2. Configure reverse proxy (Nginx)
3. Set up SSL certificates
4. Configure environment variables

### Production Frontend
1. Build production bundle: `npm run build`
2. Serve static files with web server
3. Configure API endpoint URLs
4. Set up CDN for assets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow existing code style and patterns
- Add proper error handling and user feedback
- Test on multiple browsers and devices
- Update documentation as needed
- Include medical disclaimers where appropriate

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for educational and research purposes only. The AI analysis results should not be used for medical diagnosis without consulting a qualified healthcare professional. Always seek proper medical advice for health-related concerns.

## 📞 Support

For support, questions, or contributions:
- Create an issue in the repository
- Check the [Development Guide](DEVELOPMENT_GUIDE.md)
- Review the [Frontend README](frontend/README.md)

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning capabilities
- React community for excellent frontend tools
- Medical imaging research community
- Open source contributors and maintainers
