# Medical Image Analysis System - Development Guide

This guide will help you set up and run both the backend and frontend of the Medical Image Analysis System.

## Quick Start

### 1. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```bash
   python app.py
   ```

   The backend will be available at `http://localhost:5000`

### 2. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3000`

## System Architecture

```
Medical Image Analysis System
├── backend/                 # Flask API server
│   ├── app.py              # Main Flask application
│   ├── models/             # AI model definitions
│   ├── utils/              # Utility functions
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── App.js         # Main application
│   │   └── App.css        # Global styles
│   └── package.json       # Node.js dependencies
└── notebooks/             # Jupyter notebooks for training
```

## API Endpoints

### Main Endpoints
- `POST /api/predict` - Analyze brain scan image
- `GET /api/health` - Health check
- `GET /api/classes` - Available tumor classes
- `GET /api/model/info` - Model information

### Debug Endpoints
- `POST /api/debug/prediction` - Detailed prediction analysis
- `POST /api/debug/class-order` - Test different class orders
- `GET /api/analytics/summary` - Prediction analytics
- `GET /api/predictions/history` - Recent predictions

## Features

### Backend Features
- ✅ Flask REST API with CORS support
- ✅ AI model integration (TensorFlow/Keras)
- ✅ Image validation and preprocessing
- ✅ Database integration for prediction history
- ✅ Rate limiting and error handling
- ✅ Comprehensive logging and debugging

### Frontend Features
- ✅ Modern React interface
- ✅ Drag-and-drop image upload
- ✅ Real-time analysis results
- ✅ Responsive design
- ✅ Error handling and user feedback
- ✅ Medical disclaimer and warnings

## Development Workflow

### 1. Backend Development
- Modify `backend/app.py` for API changes
- Update `backend/models/` for model changes
- Test endpoints using the built-in test page at `http://localhost:5000/test`

### 2. Frontend Development
- Modify components in `frontend/src/components/`
- Update styles in corresponding `.css` files
- Test in browser with hot reload enabled

### 3. Testing
- Backend: Use the test page or Postman
- Frontend: Use browser developer tools
- Integration: Test full workflow from upload to results

## Troubleshooting

### Common Issues

1. **Backend not starting**
   - Check if port 5000 is available
   - Verify Python dependencies are installed
   - Check for model file in `trained_models/` directory

2. **Frontend not connecting to backend**
   - Ensure backend is running on port 5000
   - Check browser console for CORS errors
   - Verify API_BASE_URL in ImageUpload.js

3. **Image upload failing**
   - Check file size (max 10MB)
   - Verify file format (JPEG/PNG only)
   - Check network connectivity

4. **Analysis results not displaying**
   - Check browser console for errors
   - Verify backend response format
   - Test with debug endpoints

### Debug Mode

Enable debug mode by using the debug endpoints:
- `/api/debug/prediction` - See raw model output
- `/api/debug/class-order` - Test different interpretations
- `/api/debug/database` - Check database connectivity

## Production Deployment

### Backend
1. Set up production WSGI server (Gunicorn)
2. Configure reverse proxy (Nginx)
3. Set up SSL certificates
4. Configure environment variables

### Frontend
1. Build production bundle: `npm run build`
2. Serve static files with web server
3. Configure API endpoint URLs
4. Set up CDN for assets

## Contributing

1. Follow existing code style and patterns
2. Add proper error handling
3. Include user feedback for all operations
4. Test on multiple browsers and devices
5. Update documentation as needed

## Security Considerations

- Images are processed client-side before upload
- No sensitive data is stored in browser
- API includes rate limiting
- Medical disclaimer is prominently displayed
- Results should not be used for medical diagnosis

## Performance Optimization

- Images are resized to 224x224 for analysis
- Frontend uses lazy loading for components
- API responses are cached where appropriate
- Database queries are optimized
- Static assets are compressed

## Monitoring

- Backend logs all predictions and errors
- Frontend includes error boundary components
- API health check endpoint available
- Database analytics endpoint for insights
