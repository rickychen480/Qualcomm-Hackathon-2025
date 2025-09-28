CSS = """
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
    }
    
    .threat-alert {
        background: linear-gradient(90deg, #ff4444, #cc0000);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
        animation: pulse 1s linear infinite;
        margin: 1rem 0;
        border: 2px solid #ff6666;
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        min-width: 300px;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-inactive {
        color: #dc3545;
        font-weight: bold;
    }
    
    .camera-feed {
        background: linear-gradient(45deg, #000, #1a1a1a);
        color: #00ff00;
        padding: 3rem 2rem;
        border-radius: 8px;
        text-align: center;
        font-family: 'Courier New', monospace;
        border: 2px solid #28a745;
        position: relative;
        overflow: hidden;
    }
    
    .camera-feed::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 0, 0.1), transparent);
        animation: scan 2s linear infinite;
    }
    
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .audio-bar-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .archive-item {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .archive-item:hover {
        background: #f8f9fa;
        border-color: #007bff;
    }
    
    .detection-running {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    .detection-stopped {
        background: #6c757d;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
    }
</style>
"""

MAIN_HEADER = """
<div class="main-header">
    <h1>HomeEdge Security Assistant</h1>
    <p>AI-Powered Edge Security for Snapdragon X Processors</p>
    <small>Real-time Threat Detection • Local Processing • Privacy First</small>
</div>
"""

METRIC_CARD = lambda alert: f"""
<div class="metric-card">
    <p><strong>Time:</strong> {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Type:</strong> {alert['type'].title()}</p>
    <p><strong>Confidence:</strong> {alert['confidence']:.1%}</p>
    <p><strong>Details:</strong> {len(alert.get('details', []))} object(s) detected</p>
</div>
"""

STATUS_CLASS = lambda status_class, status_text: f'<div class="{status_class}">{status_text}</div>'

THREAT_ALERT = lambda alert: f"""
<div class="threat-alert" id="popup-alert">
    <h3>THREAT DETECTED</h3>
    <p><strong>Type:</strong> {alert['type'].upper()}</p>
    <p><strong>Confidence:</strong> {alert['confidence']:.1%}</p>
    <p><strong>Time:</strong> {alert['timestamp'].strftime('%H:%M:%S')}</p>
    <button onclick="document.getElementById('popup-alert').style.display='none'">
        Close Alert
    </button>
</div>
"""