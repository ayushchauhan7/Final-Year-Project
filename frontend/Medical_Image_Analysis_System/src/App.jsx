import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

const API_BASE_URL = 'http://localhost:5000';

// Tumor Information Helper Function
const getTumorInformation = (prediction, confidence) => {
  const confidenceValue = parseFloat(confidence);
  
  const tumorDescriptions = {
    'glioma': {
      name: 'Glioma Tumor',
      description: 'Gliomas are tumors that originate from glial cells in the brain or spine. They are the most common type of primary brain tumors.',
      severity: 'High Risk',
      color: 'danger',
      icon: '⚠️',
      details: [
        'Most common primary brain tumor in adults',
        'Can be slow-growing (low-grade) or fast-growing (high-grade)',
        'May cause headaches, seizures, and neurological symptoms',
        'Treatment options include surgery, radiation, and chemotherapy'
      ]
    },
    'meningioma': {
      name: 'Meningioma Tumor',
      description: 'Meningiomas develop from the meninges (protective membranes covering the brain and spinal cord). Most are benign and slow-growing.',
      severity: 'Moderate Risk',
      color: 'warning',
      icon: '⚡',
      details: [
        'Usually benign (non-cancerous) and slow-growing',
        'More common in women than men',
        'May not require immediate treatment if small',
        'Symptoms depend on tumor location and size',
        'Treatment includes observation, surgery, or radiation'
      ]
    },
    'pituitary': {
      name: 'Pituitary Tumor',
      description: 'Pituitary tumors form in the pituitary gland at the base of the brain. Most are benign adenomas that can affect hormone production.',
      severity: 'Moderate Risk',
      color: 'info',
      icon: '🔬',
      details: [
        'Usually benign (non-cancerous)',
        'Can affect hormone levels and bodily functions',
        'May cause vision problems if pressing on optic nerves',
        'Symptoms include headaches, fatigue, and hormonal changes',
        'Treatment includes medication, surgery, or radiation'
      ]
    },
    'notumor': {
      name: 'No Tumor Detected',
      description: 'The AI analysis indicates no signs of tumor in the MRI scan. The brain tissue appears normal.',
      severity: 'Low Risk',
      color: 'success',
      icon: '✅',
      details: [
        'No abnormal growth detected',
        'Brain tissue appears within normal parameters',
        'Continue regular health monitoring',
        'Consult healthcare provider for any symptoms'
      ]
    }
  };
  
  // Determine tumor type
  let tumorType = 'notumor';
  if (prediction.toLowerCase().includes('glioma')) {
    tumorType = 'glioma';
  } else if (prediction.toLowerCase().includes('meningioma')) {
    tumorType = 'meningioma';
  } else if (prediction.toLowerCase().includes('pituitary')) {
    tumorType = 'pituitary';
  }
  
  const info = tumorDescriptions[tumorType];
  
  // Confidence-based recommendations
  let confidenceLevel = '';
  let recommendations = [];
  
  if (confidenceValue >= 90) {
    confidenceLevel = 'Very High Confidence';
    if (tumorType !== 'notumor') {
      recommendations = [
        '🏥 Immediate Action Required: Schedule urgent consultation with a neurologist or neurosurgeon',
        '📋 Bring complete medical history and all previous scans to the appointment',
        '🔬 Additional diagnostic tests (biopsy, advanced imaging) may be recommended',
        '👨‍⚕️ Seek second opinion from specialized brain tumor center',
        '📱 Do not delay - early intervention improves treatment outcomes'
      ];
    } else {
      recommendations = [
        '✅ No Immediate Concerns: Results indicate healthy brain tissue',
        '📅 Continue routine health check-ups as per your doctor\'s schedule',
        '🧠 Maintain brain health through proper diet, exercise, and sleep',
        '⚠️ Monitor for any new symptoms (headaches, vision changes, seizures)',
        '📞 Contact healthcare provider if any concerns arise'
      ];
    }
  } else if (confidenceValue >= 70) {
    confidenceLevel = 'High Confidence';
    if (tumorType !== 'notumor') {
      recommendations = [
        '🏥 Medical Consultation Strongly Advised: See a neurologist within 1-2 weeks',
        '📋 Request additional MRI sequences or CT scan for confirmation',
        '🔬 Prepare questions about treatment options and next steps',
        '📊 Compare with previous scans if available',
        '👨‍⚕️ Consider consultation at specialized center'
      ];
    } else {
      recommendations = [
        '✅ Likely Normal: Results suggest healthy brain tissue',
        '📅 Discuss results with your primary care physician',
        '🔍 Additional imaging may be recommended if symptoms persist',
        '📝 Keep record of this scan for future reference',
        '⚠️ Report any new or worsening symptoms immediately'
      ];
    }
  } else if (confidenceValue >= 50) {
    confidenceLevel = 'Moderate Confidence';
    recommendations = [
      '🔍 Further Investigation Needed: Results are inconclusive',
      '📋 Additional imaging with different MRI sequences recommended',
      '👨‍⚕️ Consultation with radiologist and neurologist advised',
      '🔬 Consider advanced imaging (fMRI, PET scan) if symptoms present',
      '📊 Clinical correlation with symptoms is essential',
      '⏰ Do not ignore symptoms - seek medical evaluation'
    ];
  } else {
    confidenceLevel = 'Low Confidence';
    recommendations = [
      '⚠️ Uncertain Results: AI analysis has low confidence',
      '🔄 Repeat MRI scan with optimal imaging parameters',
      '📋 Image quality may need improvement for accurate diagnosis',
      '👨‍⚕️ Professional radiologist review is essential',
      '🔬 Consider alternative imaging modalities',
      '📞 Consult healthcare provider regardless of AI results'
    ];
  }
  
  return {
    ...info,
    confidenceLevel,
    confidenceValue,
    recommendations,
    tumorType
  };
};

function App() {
  // ========== AUTHENTICATION STATE ==========
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [authMode, setAuthMode] = useState('login'); // 'login' or 'register'
  
  // Form States
  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  const [registerForm, setRegisterForm] = useState({
    username: '', email: '', password: '', fullName: ''
  });

  // ========== EXISTING STATE MANAGEMENT ==========
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [apiResponse, setApiResponse] = useState(null);
  const [debugResponse, setDebugResponse] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [history, setHistory] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [chartsData, setChartsData] = useState(null);
  const [statisticsData, setStatisticsData] = useState(null);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  // ========== AUTHENTICATION FUNCTIONS ==========
  
  // Check authentication on mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (token && userData) {
      verifyToken(token);
    }
  }, []);

  // Verify JWT token
  const verifyToken = async (token) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/verify`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.ok) {
        const data = await response.json();
        setUser(JSON.parse(localStorage.getItem('user')));
        setIsAuthenticated(true);
        loadHistoryAndAnalytics();
      } else {
        handleLogout();
      }
    } catch (err) {
      handleLogout();
    }
  };

  // Handle Login
  const handleLogin = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(loginForm)
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('token', data.token);
        localStorage.setItem('user', JSON.stringify(data.user));
        setUser(data.user);
        setIsAuthenticated(true);
        setLoginForm({ username: '', password: '' });
        loadHistoryAndAnalytics();
      } else {
        setError(data.error || 'Login failed');
      }
    } catch (err) {
      setError('Connection error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handle Register
  const handleRegister = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(registerForm)
      });

      const data = await response.json();

      if (response.ok) {
        alert('Registration successful! Please login.');
        setAuthMode('login');
        setRegisterForm({ username: '', email: '', password: '', fullName: '' });
      } else {
        setError(data.error || 'Registration failed');
      }
    } catch (err) {
      setError('Connection error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handle Logout
  const handleLogout = async () => {
    const token = localStorage.getItem('token');
    
    if (token) {
      try {
        await fetch(`${API_BASE_URL}/api/auth/logout`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${token}` }
        });
      } catch (err) {
        console.error('Logout error:', err);
      }
    }
    
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setIsAuthenticated(false);
    setHistory([]);
    setResult(null);
    setBatchResults(null);
    setAnalytics(null);
    setApiResponse(null);
    setDebugResponse(null);
  };

  // ========== EXISTING FILE HANDLING FUNCTIONS ==========
  
  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
      
      // Clear previous results
      setResult(null);
      setApiResponse(null);
      setDebugResponse(null);
    }
  };

  // Handle multiple file selection
  const handleMultipleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    setBatchResults(null);
  };

  // ========== PREDICTION FUNCTIONS WITH AUTH ==========
  
  // Upload and predict via API
  const handleApiPredict = async () => {
    if (!selectedFile) {
      alert('Please select an image first!');
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(`${API_BASE_URL}/api/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        },
      });
      
      const data = response.data;
      
      // Get enhanced tumor information
      const confidence = data.confidence_percentage || (data.confidence * 100);
      const tumorInfo = getTumorInformation(data.prediction, confidence);
      
      // Store enhanced result
      setApiResponse({
        ...data,
        tumorInfo
      });
      setResult(data);
      
      // Refresh history
      loadHistoryAndAnalytics();
    } catch (error) {
      console.error('Error:', error);
      if (error.response && error.response.status === 401) {
        setError('Session expired. Please login again.');
        handleLogout();
      } else {
        setApiResponse({ error: error.message });
        setError(error.response?.data?.error || error.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // Debug prediction
  const handleDebugPredict = async () => {
    if (!selectedFile) {
      alert('Please select an image first!');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(`${API_BASE_URL}/api/debug/prediction`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': token ? `Bearer ${token}` : undefined
        },
      });
      setDebugResponse(response.data);
    } catch (error) {
      console.error('Error:', error);
      setDebugResponse({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  // Handle batch prediction
  const handleBatchPredict = async () => {
    if (selectedFiles.length === 0) {
      alert('Please select multiple images first!');
      return;
    }

    setBatchLoading(true);
    setError(null);
    const formData = new FormData();
    
    selectedFiles.forEach(file => {
      formData.append('images', file);
    });

    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(`${API_BASE_URL}/api/predict/batch`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        },
      });
      setBatchResults(response.data);
      
      // Refresh history
      loadHistoryAndAnalytics();
    } catch (error) {
      console.error('Error:', error);
      if (error.response && error.response.status === 401) {
        setError('Session expired. Please login again.');
        handleLogout();
      } else {
        setBatchResults({ error: error.message });
        setError(error.response?.data?.error || error.message);
      }
    } finally {
      setBatchLoading(false);
    }
  };

  // ========== EXISTING DATA LOADING FUNCTIONS ==========
  
  // Load system information
  const loadSystemInfo = async () => {
    try {
      const [healthRes, classesRes, modelRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/health`),
        axios.get(`${API_BASE_URL}/api/classes`),
        axios.get(`${API_BASE_URL}/api/model/info`)
      ]);

      setSystemInfo({
        health: healthRes.data,
        classes: classesRes.data,
        model: modelRes.data
      });
    } catch (error) {
      console.error('Error loading system info:', error);
    }
  };

  // Load history and analytics
  const loadHistoryAndAnalytics = async () => {
    const token = localStorage.getItem('token');
    if (!token) return;

    try {
      const [historyRes, analyticsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/predictions/history?limit=10`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        axios.get(`${API_BASE_URL}/api/analytics/summary`, {
          headers: { 'Authorization': `Bearer ${token}` }
        })
      ]);

      setHistory(historyRes.data.predictions || historyRes.data.recent_predictions || []);
      setAnalytics(analyticsRes.data);
    } catch (error) {
      console.error('Error loading data:', error);
      if (error.response && error.response.status === 401) {
        handleLogout();
      }
    }
  };

  // Load charts
  const loadCharts = async () => {
    setChartsLoading(true);
    try {
      const [chartsRes, statsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/results/charts`),
        axios.get(`${API_BASE_URL}/api/results/statistics`)
      ]);
      
      setChartsData(chartsRes.data);
      setStatisticsData(statsRes.data);
    } catch (error) {
      console.error('Error loading charts:', error);
    } finally {
      setChartsLoading(false);
    }
  };

  // Clear all data
  const clearAll = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setResult(null);
    setApiResponse(null);
    setDebugResponse(null);
    setSelectedFiles([]);
    setBatchResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // ========== LOGIN/REGISTER UI ==========
  
  if (!isAuthenticated) {
    return (
      <div className="auth-gradient-bg">
        <div className="auth-card">
          {/* Header */}
          <div className="auth-card-header">
            <div className="brain-icon">🧠</div>
            <h2>Brain Tumor Detection</h2>
            <p>AI-Powered Medical Image Analysis System</p>
          </div>

          {/* Body */}
          <div className="auth-card-body">
            {/* Tabs */}
            <div className="auth-tabs">
              <button
                className={`auth-tab-btn ${authMode === 'login' ? 'active' : ''}`}
                onClick={() => setAuthMode('login')}
              >
                Login
              </button>
              <button
                className={`auth-tab-btn ${authMode === 'register' ? 'active' : ''}`}
                onClick={() => setAuthMode('register')}
              >
                Register
              </button>
            </div>

            {/* Error Alert */}
            {error && (
              <div className="alert-error">
                ⚠️ {error}
              </div>
            )}

            {/* Login Form */}
            {authMode === 'login' ? (
              <form onSubmit={handleLogin}>
                <div className="form-group">
                  <label className="form-label">Username</label>
                  <input
                    type="text"
                    className="form-input"
                    value={loginForm.username}
                    onChange={(e) => setLoginForm({...loginForm, username: e.target.value})}
                    placeholder="Enter your username"
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Password</label>
                  <input
                    type="password"
                    className="form-input"
                    value={loginForm.password}
                    onChange={(e) => setLoginForm({...loginForm, password: e.target.value})}
                    placeholder="Enter your password"
                    required
                  />
                </div>

                <button type="submit" className="btn-submit" disabled={loading}>
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Logging In...
                    </>
                  ) : (
                    'Login'
                  )}
                </button>
              </form>
            ) : (
              /* Register Form */
              <form onSubmit={handleRegister}>
                <div className="form-group">
                  <label className="form-label">Full Name</label>
                  <input
                    type="text"
                    className="form-input"
                    value={registerForm.fullName}
                    onChange={(e) => setRegisterForm({...registerForm, fullName: e.target.value})}
                    placeholder="Enter your full name"
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Username</label>
                  <input
                    type="text"
                    className="form-input"
                    value={registerForm.username}
                    onChange={(e) => setRegisterForm({...registerForm, username: e.target.value})}
                    placeholder="Choose a username"
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Email</label>
                  <input
                    type="email"
                    className="form-input"
                    value={registerForm.email}
                    onChange={(e) => setRegisterForm({...registerForm, email: e.target.value})}
                    placeholder="Enter your email"
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Password</label>
                  <input
                    type="password"
                    className="form-input"
                    value={registerForm.password}
                    onChange={(e) => setRegisterForm({...registerForm, password: e.target.value})}
                    placeholder="Create a strong password"
                    required
                    minLength="6"
                  />
                </div>

                <button type="submit" className="btn-submit" disabled={loading}>
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Creating Account...
                    </>
                  ) : (
                    'Register'
                  )}
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ========== MAIN APPLICATION UI (AUTHENTICATED) ==========
  
  return (
    <div className="container-fluid">
      {/* Header with User Info */}
      <div className="row bg-primary text-white py-4 mb-4">
        <div className="col-12">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="display-4 mb-2">🧠 Medical Image Analysis System</h1>
              <p className="lead mb-0">Advanced Brain Tumor Detection with AI</p>
            </div>
            <div className="text-end">
              <div className="mb-2">
                <strong>{user?.fullName}</strong><br/>
                <small>@{user?.username}</small>
              </div>
              <button onClick={handleLogout} className="btn btn-outline-light btn-sm">
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="row mb-4">
        <div className="col-12">
          <ul className="nav nav-pills nav-fill">
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'upload' ? 'active' : ''}`}
                onClick={() => setActiveTab('upload')}
              >
                🔬 Image Analysis
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'batch' ? 'active' : ''}`}
                onClick={() => setActiveTab('batch')}
              >
                📚 Batch Analysis
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'results' ? 'active' : ''}`}
                onClick={() => { setActiveTab('results'); loadCharts(); }}
              >
                📊 Results & Analytics
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'system' ? 'active' : ''}`}
                onClick={() => { setActiveTab('system'); loadSystemInfo(); }}
              >
                ⚙️ System Info
              </button>
            </li>
            <li className="nav-item">
              <button 
                className={`nav-link ${activeTab === 'history' ? 'active' : ''}`}
                onClick={() => { setActiveTab('history'); loadHistoryAndAnalytics(); }}
              >
                📊 Analytics
              </button>
            </li>
          </ul>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="alert alert-danger alert-dismissible fade show" role="alert">
          {error}
          <button type="button" className="btn-close" onClick={() => setError(null)}></button>
        </div>
      )}

      {/* Main Content */}
      <div className="row">
        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="col-12">
            <div className="row">
              {/* Upload Section */}
              <div className="col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-header bg-info text-white">
                    <h5 className="mb-0">📤 Upload MRI Scan</h5>
                  </div>
                  <div className="card-body">
                    <div className="mb-3">
                      <label className="form-label">Select Brain MRI Image:</label>
                      <input
                        type="file"
                        className="form-control"
                        accept="image/*"
                        onChange={handleFileSelect}
                        ref={fileInputRef}
                      />
                    </div>

                    {imagePreview && (
                      <div className="mb-3">
                        <h6>Image Preview:</h6>
                        <img 
                          src={imagePreview} 
                          alt="Preview" 
                          className="img-fluid rounded border"
                          style={{maxHeight: '300px'}}
                        />
                      </div>
                    )}

                    <div className="d-grid gap-2">
                      <button 
                        className="btn btn-primary"
                        onClick={handleApiPredict}
                        disabled={loading || !selectedFile}
                      >
                        {loading ? (
                          <>
                            <span className="spinner-border spinner-border-sm me-2"></span>
                            Analyzing...
                          </>
                        ) : (
                          '🔍 Analyze Image'
                        )}
                      </button>
                      
                      <button 
                        className="btn btn-success"
                        onClick={handleDebugPredict}
                        disabled={loading || !selectedFile}
                      >
                        🔬 Debug Analysis
                      </button>
                      
                      <button 
                        className="btn btn-secondary"
                        onClick={clearAll}
                      >
                        🗑️ Clear All
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Results Section with Enhanced Medical Information */}
              <div className="col-md-6">
                {apiResponse && apiResponse.tumorInfo && (
                  <div className="card shadow-sm mb-3">
                    {/* Primary Result Card */}
                    <div className={`card-header bg-${apiResponse.tumorInfo.color} text-white`}>
                      <h5 className="mb-0">{apiResponse.tumorInfo.icon} Analysis Result</h5>
                    </div>
                    <div className="card-body">
                      <h4 className={`text-${apiResponse.tumorInfo.color} mb-2`}>
                        {apiResponse.tumorInfo.name}
                      </h4>
                      
                      <div className="row mb-3">
                        <div className="col-6">
                          <div className="alert alert-light border p-2">
                            <small className="text-muted d-block">Confidence Level</small>
                            <strong className={`text-${apiResponse.tumorInfo.color}`}>
                              {apiResponse.confidence_percentage?.toFixed(2) || apiResponse.confidence}%
                            </strong>
                            <br />
                            <small className="text-muted">
                              {apiResponse.tumorInfo.confidenceLevel}
                            </small>
                          </div>
                        </div>
                        <div className="col-6">
                          <div className="alert alert-light border p-2">
                            <small className="text-muted d-block">Severity</small>
                            <strong className={`text-${apiResponse.tumorInfo.color}`}>
                              {apiResponse.tumorInfo.severity}
                            </strong>
                          </div>
                        </div>
                      </div>
                      
                      <p className="mb-0">
                        <small>{apiResponse.tumorInfo.description}</small>
                      </p>
                    </div>
                    
                    {/* Medical Information Card */}
                    <div className="card-header bg-info text-white">
                      <strong>📋 Medical Information</strong>
                    </div>
                    <div className="card-body">
                      <ul style={{fontSize: '0.85rem'}}>
                        {apiResponse.tumorInfo.details.map((detail, idx) => (
                          <li key={idx}>{detail}</li>
                        ))}
                      </ul>
                    </div>
                    
                    {/* Recommendations Card */}
                    <div className={`card-header bg-${apiResponse.tumorInfo.color} text-white`}>
                      <strong>💡 Recommendations</strong>
                    </div>
                    <div className="card-body">
                      <ol style={{fontSize: '0.85rem'}}>
                        {apiResponse.tumorInfo.recommendations.map((rec, idx) => (
                          <li key={idx}>{rec}</li>
                        ))}
                      </ol>
                    </div>
                    
                    {/* Disclaimer */}
                    <div className="card-footer">
                      <div className="alert alert-warning mb-0">
                        <strong>⚠️ Important Disclaimer:</strong> This AI analysis is a screening tool and NOT a definitive diagnosis. 
                        Always consult qualified healthcare professionals for proper medical evaluation and treatment decisions.
                      </div>
                    </div>
                  </div>
                )}

                {/* Debug Results */}
                {debugResponse && (
                  <div className="card shadow-sm">
                    <div className="card-header bg-warning text-dark">
                      <h5 className="mb-0">🔬 Debug Information</h5>
                    </div>
                    <div className="card-body">
                      <pre className="bg-light p-3 rounded" style={{fontSize: '12px', maxHeight: '400px', overflow: 'auto'}}>
                        {JSON.stringify(debugResponse, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Batch Analysis Tab */}
        {activeTab === 'batch' && (
          <div className="col-12">
            <div className="row">
              {/* Batch Upload Section */}
              <div className="col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-header bg-warning text-dark">
                    <h5 className="mb-0">📚 Multiple Image Upload</h5>
                  </div>
                  <div className="card-body">
                    <div className="mb-3">
                      <label className="form-label">Select Multiple MRI Images:</label>
                      <input
                        type="file"
                        className="form-control"
                        accept="image/*"
                        multiple
                        onChange={handleMultipleFileSelect}
                      />
                      <small className="text-muted">You can select multiple images at once</small>
                    </div>

                    {selectedFiles.length > 0 && (
                      <div className="mb-3">
                        <h6>Selected Images ({selectedFiles.length}):</h6>
                        <div className="row">
                          {selectedFiles.slice(0, 4).map((file, index) => (
                            <div key={index} className="col-6 col-md-3 mb-2">
                              <div className="border rounded p-2 text-center">
                                <small className="text-truncate d-block">{file.name}</small>
                                <small className="text-muted">{(file.size / 1024).toFixed(1)} KB</small>
                              </div>
                            </div>
                          ))}
                          {selectedFiles.length > 4 && (
                            <div className="col-12">
                              <small className="text-muted">... and {selectedFiles.length - 4} more images</small>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="d-grid gap-2">
                      <button 
                        className="btn btn-warning"
                        onClick={handleBatchPredict}
                        disabled={batchLoading || selectedFiles.length === 0}
                      >
                        {batchLoading ? (
                          <>
                            <span className="spinner-border spinner-border-sm me-2"></span>
                            Processing {selectedFiles.length} images...
                          </>
                        ) : (
                          `🔍 Analyze ${selectedFiles.length} Images`
                        )}
                      </button>
                      
                      <button 
                        className="btn btn-secondary"
                        onClick={() => {setSelectedFiles([]); setBatchResults(null);}}
                      >
                        🗑️ Clear Selection
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Batch Results Section with Medical Information */}
              <div className="col-md-6">
                {batchResults && (
                  <div className="card shadow-sm">
                    <div className="card-header bg-info text-white">
                      <h5 className="mb-0">📊 Batch Analysis Results</h5>
                    </div>
                    <div className="card-body">
                      {batchResults.error ? (
                        <div className="alert alert-danger">
                          <strong>Error:</strong> {batchResults.error}
                        </div>
                      ) : (
                        <>
                          {/* Batch Summary */}
                          <div className="mb-3 p-3 bg-light rounded">
                            <h6>📈 Batch Summary</h6>
                            <div className="row text-center">
                              <div className="col-4">
                                <strong className="text-primary">{batchResults.total_images}</strong><br/>
                                <small>Total Images</small>
                              </div>
                              <div className="col-4">
                                <strong className="text-danger">{batchResults.batch_summary?.tumor_detected || 0}</strong><br/>
                                <small>Tumors Found</small>
                              </div>
                              <div className="col-4">
                                <strong className="text-success">{batchResults.batch_summary?.no_tumor || 0}</strong><br/>
                                <small>Normal Scans</small>
                              </div>
                            </div>
                            
                            {/* Tumor Type Breakdown */}
                            {batchResults.batch_summary?.by_tumor_type && Object.keys(batchResults.batch_summary.by_tumor_type).length > 0 && (
                              <div className="mt-3">
                                <small className="text-muted">Tumor Type Distribution:</small>
                                <div className="row mt-2">
                                  {Object.entries(batchResults.batch_summary.by_tumor_type).map(([type, count]) => (
                                    <div key={type} className="col-6 col-md-3 mb-2">
                                      <div className="text-center p-2 border rounded">
                                        <strong>{count}</strong><br/>
                                        <small>{type}</small>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Enhanced Image Results with Medical Info */}
                          <div style={{maxHeight: '600px', overflow: 'auto'}}>
                            <h6>📋 Detailed Results:</h6>
                            {batchResults.results && batchResults.results.map((result, index) => {
                              const file = selectedFiles.find(f => f.name === result.filename);
                              const confidence = result.confidence_percentage || (result.confidence_score * 100);
                              const tumorInfo = getTumorInformation(result.prediction, confidence);
                              
                              return (
                                <div key={index} className="card mb-3">
                                  <div className="row g-0">
                                    {/* Image Preview */}
                                    <div className="col-md-4">
                                      {file && (
                                        <img 
                                          src={URL.createObjectURL(file)}
                                          alt={result.filename}
                                          className="img-fluid rounded-start"
                                          style={{
                                            width: '100%', 
                                            height: '200px', 
                                            objectFit: 'cover'
                                          }}
                                        />
                                      )}
                                      <div className="p-2">
                                        <small className="text-muted text-truncate d-block" title={result.filename}>
                                          📄 {result.filename}
                                        </small>
                                        {file && (
                                          <small className="text-muted">
                                            💾 {(file.size / 1024).toFixed(1)} KB
                                          </small>
                                        )}
                                      </div>
                                    </div>
                                    
                                    {/* Medical Information */}
                                    <div className="col-md-8">
                                      <div className="card-body p-2">
                                        {/* Result Header */}
                                        <div className={`alert alert-${tumorInfo.color} mb-2 p-2`}>
                                          <div className="d-flex justify-content-between align-items-center">
                                            <span>
                                              <strong>{tumorInfo.icon} {tumorInfo.name}</strong>
                                            </span>
                                            <span className={`badge bg-${tumorInfo.color}`}>
                                              {tumorInfo.severity}
                                            </span>
                                          </div>
                                          <small className="d-block mt-1">{tumorInfo.description}</small>
                                        </div>
                                        
                                        {/* Confidence & Metrics */}
                                        <div className="row mb-2">
                                          <div className="col-6">
                                            <div className="border rounded p-2 text-center">
                                              <small className="text-muted d-block">Confidence</small>
                                              <strong className={`text-${tumorInfo.color}`}>
                                                {typeof confidence === 'number' ? confidence.toFixed(1) : confidence}%
                                              </strong>
                                              <br/>
                                              <small className="text-muted">{tumorInfo.confidenceLevel}</small>
                                            </div>
                                          </div>
                                          <div className="col-6">
                                            <div className="border rounded p-2">
                                              <small className="text-muted d-block">Processing Time</small>
                                              <strong>{result.processing_time || 'N/A'}</strong>
                                            </div>
                                          </div>
                                        </div>
                                        
                                        {/* Compact Medical Details */}
                                        <details className="mb-2">
                                          <summary className="btn btn-sm btn-outline-info w-100" style={{cursor: 'pointer'}}>
                                            📋 Medical Information
                                          </summary>
                                          <ul className="mt-2 mb-0" style={{fontSize: '0.75rem'}}>
                                            {tumorInfo.details.map((detail, idx) => (
                                              <li key={idx}>{detail}</li>
                                            ))}
                                          </ul>
                                        </details>
                                        
                                        <details>
                                          <summary className="btn btn-sm btn-outline-warning w-100" style={{cursor: 'pointer'}}>
                                            💡 Recommendations
                                          </summary>
                                          <ol className="mt-2 mb-0" style={{fontSize: '0.75rem', paddingLeft: '1.2rem'}}>
                                            {tumorInfo.recommendations.slice(0, 3).map((rec, idx) => (
                                              <li key={idx}>{rec}</li>
                                            ))}
                                          </ol>
                                        </details>
                                        
                                        {/* Quick Action Badges */}
                                        <div className="mt-2">
                                          {tumorInfo.tumorType !== 'notumor' ? (
                                            <span className="badge bg-danger me-1" style={{fontSize: '0.7rem'}}>
                                              🏥 Requires Medical Attention
                                            </span>
                                          ) : (
                                            <span className="badge bg-success me-1" style={{fontSize: '0.7rem'}}>
                                              ✅ Normal Scan
                                            </span>
                                          )}
                                          <span className={`badge bg-${tumorInfo.color}`} style={{fontSize: '0.7rem'}}>
                                            {tumorInfo.confidenceLevel}
                                          </span>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                          
                          {/* Overall Disclaimer */}
                          <div className="alert alert-warning mt-3 mb-0">
                            <small>
                              <strong>⚠️ Batch Analysis Disclaimer:</strong> All results are AI-generated screening outputs. 
                              Professional radiologist review is required for clinical diagnosis and treatment planning.
                            </small>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Results & Analytics Tab */}
        {activeTab === 'results' && (
          <div className="col-12">
            {chartsLoading ? (
              <div className="text-center">
                <div className="spinner-border" role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
                <p className="mt-2">Generating charts and statistics...</p>
              </div>
            ) : (
              <div className="row">
                {/* Charts Section */}
                {chartsData && chartsData.charts && (
                  <div className="col-12 mb-4">
                    <div className="card shadow-sm">
                      <div className="card-header bg-info text-white">
                        <h5 className="mb-0">📊 Visualization Charts</h5>
                      </div>
                      <div className="card-body">
                        <div className="row">
                          {Object.entries(chartsData.charts).map(([chartName, chartData]) => (
                            <div key={chartName} className="col-md-6 mb-3">
                              <h6 className="text-center">{chartName.replace(/_/g, ' ').toUpperCase()}</h6>
                              <img 
                                src={`data:image/png;base64,${chartData}`}
                                alt={chartName}
                                className="img-fluid"
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Statistics Section */}
                {statisticsData && (
                  <div className="col-12">
                    <div className="card shadow-sm">
                      <div className="card-header bg-success text-white">
                        <h5 className="mb-0">📈 Statistics</h5>
                      </div>
                      <div className="card-body">
                        <pre>{JSON.stringify(statisticsData, null, 2)}</pre>
                      </div>
                    </div>
                  </div>
                )}

                {!chartsData && !statisticsData && (
                  <div className="col-12">
                    <div className="alert alert-info">
                      <strong>ℹ️ No Data Available:</strong> Perform some predictions first to see charts and statistics.
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* System Info Tab */}
        {activeTab === 'system' && (
          <div className="col-12">
            {systemInfo ? (
              <div className="row">
                <div className="col-md-4 mb-3">
                  <div className="card shadow-sm">
                    <div className="card-header bg-primary text-white">
                      <h5 className="mb-0">🔋 System Health</h5>
                    </div>
                    <div className="card-body">
                      <pre>{JSON.stringify(systemInfo.health, null, 2)}</pre>
                    </div>
                  </div>
                </div>
                <div className="col-md-4 mb-3">
                  <div className="card shadow-sm">
                    <div className="card-header bg-info text-white">
                      <h5 className="mb-0">🏷️ Available Classes</h5>
                    </div>
                    <div className="card-body">
                      <pre>{JSON.stringify(systemInfo.classes, null, 2)}</pre>
                    </div>
                  </div>
                </div>
                <div className="col-md-4 mb-3">
                  <div className="card shadow-sm">
                    <div className="card-header bg-success text-white">
                      <h5 className="mb-0">🤖 Model Information</h5>
                    </div>
                    <div className="card-body">
                      <pre>{JSON.stringify(systemInfo.model, null, 2)}</pre>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center">
                <div className="spinner-border" role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
                <p className="mt-2">Loading system information...</p>
              </div>
            )}
          </div>
        )}

        {/* History & Analytics Tab */}
        {activeTab === 'history' && (
          <div className="col-12">
            <div className="row">
              <div className="col-md-6 mb-3">
                <div className="card shadow-sm">
                  <div className="card-header bg-info text-white">
                    <h5 className="mb-0">📊 Analytics Summary</h5>
                  </div>
                  <div className="card-body">
                    {analytics ? (
                      <pre>{JSON.stringify(analytics, null, 2)}</pre>
                    ) : (
                      <p>No analytics data available</p>
                    )}
                  </div>
                </div>
              </div>
              <div className="col-md-6 mb-3">
                <div className="card shadow-sm">
                  <div className="card-header bg-warning text-dark">
                    <h5 className="mb-0">📝 Recent Predictions</h5>
                  </div>
                  <div className="card-body">
                    {history.length > 0 ? (
                      <div style={{maxHeight: '400px', overflow: 'auto'}}>
                        <div className="table-responsive">
                          <table className="table table-hover table-sm">
                            <thead>
                              <tr>
                                <th>Date</th>
                                <th>File</th>
                                <th>Result</th>
                                <th>Confidence</th>
                              </tr>
                            </thead>
                            <tbody>
                              {history.map((item, index) => (
                                <tr key={index}>
                                  <td>
                                    <small>{new Date(item.createdAt || item.timestamp).toLocaleString()}</small>
                                  </td>
                                  <td>
                                    <small className="text-truncate d-block" style={{maxWidth: '150px'}}>
                                      {item.filename}
                                    </small>
                                  </td>
                                  <td>
                                    <span className={`badge bg-${(item.result || item.prediction || '').includes('No Tumor') ? 'success' : 'danger'}`}>
                                      {item.result || item.prediction}
                                    </span>
                                  </td>
                                  <td>
                                    <small>{((item.confidence || 0) * 100).toFixed(1)}%</small>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ) : (
                      <p className="text-center text-muted">No prediction history available</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
