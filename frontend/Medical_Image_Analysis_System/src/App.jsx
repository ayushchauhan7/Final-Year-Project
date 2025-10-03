import React, { useState, useRef } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

const API_BASE_URL = 'http://localhost:5000';

function App() {
  // State management
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
  
  const fileInputRef = useRef(null);

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

  // Upload and predict via API
  const handleApiPredict = async () => {
    if (!selectedFile) {
      alert('Please select an image first!');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setApiResponse(response.data);
      setResult(response.data);
    } catch (error) {
      console.error('Error:', error);
      setApiResponse({ error: error.message });
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
      const response = await axios.post(`${API_BASE_URL}/api/debug/prediction`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
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
    const formData = new FormData();
    
    selectedFiles.forEach(file => {
      formData.append('images', file);
    });

    try {
      const response = await axios.post(`${API_BASE_URL}/api/predict/batch`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setBatchResults(response.data);
    } catch (error) {
      console.error('Error:', error);
      setBatchResults({ error: error.message });
    } finally {
      setBatchLoading(false);
    }
  };

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
    try {
      const [historyRes, analyticsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/predictions/history?limit=10`),
        axios.get(`${API_BASE_URL}/api/analytics/summary`)
      ]);

      setHistory(historyRes.data.recent_predictions || []);
      setAnalytics(analyticsRes.data);
    } catch (error) {
      console.error('Error loading data:', error);
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
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="container-fluid">
      {/* Header */}
      <div className="row bg-primary text-white py-4 mb-4">
        <div className="col-12 text-center">
          <h1 className="display-4 mb-2">🧠 Medical Image Analysis System</h1>
          <p className="lead">Advanced Brain Tumor Detection with AI</p>
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
            {/* Add new tab for batch processing */}
            <li className="nav-item" role="presentation">
              <button 
                className={`nav-link ${activeTab === 'batch' ? 'active' : ''}`}
                onClick={() => setActiveTab('batch')}
              >
                📚 Batch Analysis
              </button>
            </li>
            {/* Add new tab for results and analytics */}
            <li className="nav-item" role="presentation">
              <button 
                className={`nav-link ${activeTab === 'results' ? 'active' : ''}`}
                onClick={() => { setActiveTab('results'); loadCharts(); }}
              >
                📊 Results & Analytics
              </button>
            </li>
          </ul>
        </div>
      </div>

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

              {/* Results Section */}
              <div className="col-md-6">
                {/* Standard Results */}
                {apiResponse && (
                  <div className="card shadow-sm mb-3">
                    <div className="card-header bg-success text-white">
                      <h5 className="mb-0">🎯 Analysis Result</h5>
                    </div>
                    <div className="card-body">
                      {apiResponse.error ? (
                        <div className="alert alert-danger">
                          <strong>Error:</strong> {apiResponse.error}
                        </div>
                      ) : (
                        <>
                          <h4 className="text-primary">{apiResponse.prediction}</h4>
                          <p className="text-muted">Confidence: {apiResponse.confidence}</p>
                          
                          {apiResponse.probabilities && (
                            <div className="mt-3">
                              <h6>Class Probabilities:</h6>
                              {Object.entries(apiResponse.probabilities).map(([className, prob]) => (
                                <div key={className} className="mb-2">
                                  <div className="d-flex justify-content-between">
                                    <span className="text-capitalize">{className}:</span>
                                    <span>{(prob * 100).toFixed(2)}%</span>
                                  </div>
                                  <div className="progress" style={{height: '6px'}}>
                                    <div 
                                      className="progress-bar" 
                                      style={{width: `${prob * 100}%`}}
                                    ></div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </>
                      )}
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
                        {history.map((item, index) => (
                          <div key={index} className="border-bottom pb-2 mb-2">
                            <strong>{item.result}</strong><br/>
                            <small className="text-muted">
                              {item.filename} - {item.confidence}% confidence<br/>
                              {new Date(item.timestamp).toLocaleString()}
                            </small>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p>No prediction history available</p>
                    )}
                  </div>
                </div>
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

              {/* Enhanced Batch Results Section */}
              <div className="col-md-6">
                {batchResults && (
                  <div className="card shadow-sm">
                    <div className="card-header bg-info text-white">
                      <h5 className="mb-0">📊 Batch Results</h5>
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
                                <strong className="text-danger">{batchResults.batch_summary.tumor_detected}</strong><br/>
                                <small>Tumors Found</small>
                              </div>
                              <div className="col-4">
                                <strong className="text-success">{batchResults.batch_summary.no_tumor}</strong><br/>
                                <small>Normal Scans</small>
                              </div>
                            </div>
                          </div>

                          {/* Image Results with Previews */}
                          <div style={{maxHeight: '500px', overflow: 'auto'}}>
                            <h6>Image Results:</h6>
                            <div className="row">
                              {selectedFiles.slice(0, 6).map((file, index) => {
                                // Find corresponding result
                                const result = batchResults.results ? 
                                  batchResults.results.find(r => r.filename === file.name) : null;
                                
                                const prediction = result ? result.prediction : 'Processing...';
                                const confidence = result ? result.confidence : 'N/A';
                                const isTumor = prediction.includes('Tumor') && !prediction.includes('No Tumor');
                                
                                return (
                                  <div key={index} className="col-md-6 mb-3">
                                    <div className={`card ${isTumor ? 'border-danger' : 'border-success'}`}>
                                      <div style={{height: '120px', overflow: 'hidden'}}>
                                        <img 
                                          src={URL.createObjectURL(file)}
                                          alt={file.name}
                                          className="card-img-top"
                                          style={{
                                            width: '100%', 
                                            height: '100%', 
                                            objectFit: 'cover'
                                          }}
                                        />
                                      </div>
                                      <div className="card-body p-2">
                                        <h6 className="card-title text-truncate" style={{fontSize: '0.8rem'}} title={file.name}>
                                          {file.name}
                                        </h6>
                                        <div className="d-flex justify-content-between align-items-center">
                                          <span className={`badge ${isTumor ? 'bg-danger' : 'bg-success'}`} style={{fontSize: '0.7rem'}}>
                                            {prediction}
                                          </span>
                                        </div>
                                        <small className="text-muted d-block">
                                          Confidence: {confidence}
                                        </small>
                                        <small className="text-muted">
                                          {(file.size / 1024).toFixed(1)} KB
                                        </small>
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                            
                            {selectedFiles.length > 6 && (
                              <div className="alert alert-info">
                                <small>
                                  Showing first 6 of {selectedFiles.length} images. 
                                  All {selectedFiles.length} results are processed.
                                </small>
                              </div>
                            )}
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
                  <span className="visually-hidden">Loading charts...</span>
                </div>
                <p>Generating research charts and analytics...</p>
              </div>
            ) : chartsData ? (
              <div className="row">
                {/* Charts Display */}
                {Object.entries(chartsData.charts).map(([chartName, chartData]) => (
                  <div key={chartName} className="col-md-6 mb-4">
                    <div className="card shadow-sm">
                      <div className="card-header">
                        <h5 className="mb-0">{chartName.replace('_', ' ').toUpperCase()}</h5>
                      </div>
                      <div className="card-body text-center">
                        <img 
                          src={`data:image/png;base64,${chartData}`} 
                          alt={chartName}
                          className="img-fluid"
                          style={{maxHeight: '400px'}}
                        />
                      </div>
                    </div>
                  </div>
                ))}
                
                {/* Statistics Summary */}
                <div className="col-12 mt-4">
                  <div className="card shadow-sm">
                    <div className="card-header bg-info text-white">
                      <h5 className="mb-0">📈 Research Statistics Summary</h5>
                    </div>
                    <div className="card-body">
                      <pre className="bg-light p-3 rounded" style={{fontSize: '12px', maxHeight: '400px', overflow: 'auto'}}>
                        {JSON.stringify(statisticsData, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center">
                <button className="btn btn-primary" onClick={loadCharts}>
                  Generate Research Charts & Analytics
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
