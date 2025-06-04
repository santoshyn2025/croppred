from flask import Flask, render_template, jsonify
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import time
import os
from threading import Thread
from flask_cors import CORS
import json
import io

app = Flask(__name__)
CORS(app)

def initialize_firebase():
    """Initialize Firebase Admin SDK with credentials from environment variable"""
    try:
        # Get Firebase credentials from environment variable
        firebase_creds = json.loads(os.environ['FIREBASE_CREDENTIALS'])
        
        # Create a file-like object from the credentials JSON
        cred = credentials.Certificate(firebase_creds)
        
        # Initialize the Firebase app
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-agri-2d3da-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
        print("✅ Firebase initialized successfully")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing Firebase credentials: {str(e)}")
        raise
    except KeyError:
        print("❌ FIREBASE_CREDENTIALS environment variable not found")
        raise
    except Exception as e:
        print(f"❌ Error initializing Firebase: {str(e)}")
        raise

# Initialize Firebase
try:
    initialize_firebase()
except Exception as e:
    print(f"Failed to initialize Firebase: {e}")
# Crop Ranges
CROP_RANGES = {
    "rice":       {"N": (20, 50), "P": (30, 60), "K": (5, 15), "EC": (25, 30), "pH": (5.5, 7.0),
                   "moisture": (1.0, 2.5), "lux": (5000, 10000), "temp": (25, 35), "humidity": (60, 85)},
    "wheat":      {"N": (25, 60), "P": (40, 70), "K": (5, 15), "EC": (22, 28), "pH": (6.0, 7.5),
                   "moisture": (0.8, 1.8), "lux": (6000, 12000), "temp": (15, 25), "humidity": (50, 70)},
    "maize":      {"N": (30, 70), "P": (50, 80), "K": (5, 20), "EC": (24, 30), "pH": (5.8, 7.2),
                   "moisture": (1.0, 2.0), "lux": (7000, 15000), "temp": (20, 30), "humidity": (50, 80)},
    "potato":     {"N": (30, 60), "P": (40, 80), "K": (60, 150), "EC": (26, 32), "pH": (5.0, 6.5),
                   "moisture": (1.2, 2.2), "lux": (3000, 7000), "temp": (15, 22), "humidity": (70, 90)},
    "tomato":     {"N": (40, 70), "P": (30, 50), "K": (40, 70), "EC": (28, 32), "pH": (6.0, 6.8),
                   "moisture": (1.0, 1.5), "lux": (6000, 12000), "temp": (18, 27), "humidity": (60, 80)},
    "onion":      {"N": (25, 50), "P": (20, 40), "K": (30, 60), "EC": (20, 24), "pH": (6.0, 7.0),
                   "moisture": (0.8, 1.3), "lux": (4000, 9000), "temp": (15, 25), "humidity": (55, 75)},
    "carrot":     {"N": (30, 60), "P": (30, 50), "K": (40, 80), "EC": (22, 26), "pH": (6.0, 7.0),
                   "moisture": (1.0, 1.6), "lux": (4000, 8000), "temp": (15, 22), "humidity": (60, 80)},
    "cotton":    {"N": (50, 120), "P": (30, 70), "K": (40, 90), "EC": (26, 32), "pH": (5.5, 7.5),
                   "moisture": (1.2, 2.0), "lux": (7000, 14000), "temp": (25, 35), "humidity": (50, 80)},
    "soybean":   {"N": (40, 90), "P": (30, 60), "K": (30, 70), "EC": (28, 34), "pH": (6.0, 7.0),
                   "moisture": (1.0, 1.8), "lux": (6000, 11000), "temp": (20, 30), "humidity": (55, 75)},
    "cabbage":   {"N": (35, 60), "P": (25, 50), "K": (40, 70), "EC": (30, 36), "pH": (6.0, 7.5),
                   "moisture": (1.0, 1.8), "lux": (5000, 9000), "temp": (15, 22), "humidity": (60, 85)},
    "cauliflower": {"N": (30, 55), "P": (25, 50), "K": (40, 80), "EC": (27, 31), "pH": (6.0, 7.5),
                    "moisture": (1.1, 1.8), "lux": (5000, 9000), "temp": (15, 22), "humidity": (65, 85)},
    "sugarcane":  {"N": (60, 150), "P": (30, 70), "K": (80, 180), "EC": (25, 31), "pH": (6.0, 7.5),
                   "moisture": (1.5, 3.0), "lux": (7000, 15000), "temp": (25, 35), "humidity": (70, 90)},
    "banana":     {"N": (40, 90), "P": (25, 55), "K": (70, 130), "EC": (23, 29), "pH": (5.5, 7.0),
                   "moisture": (1.0, 2.5), "lux": (6000, 11000), "temp": (22, 32), "humidity": (70, 90)},
    "pepper":     {"N": (50, 100), "P": (30, 70), "K": (40, 90), "EC": (19, 30), "pH": (5.5, 6.5),
                   "moisture": (1.2, 2.0), "lux": (7000, 12000), "temp": (20, 30), "humidity": (70, 85)},
    "chili":      {"N": (40, 80), "P": (30, 60), "K": (40, 80), "EC": (17, 32), "pH": (6.0, 7.0),
                   "moisture": (1.0, 1.6), "lux": (6000, 11000), "temp": (20, 30), "humidity": (60, 80)}
}

# Global variable to store analysis results
latest_analysis = {
    "timestamp": "",
    "data": {},
    "recommendations": [],
    "problems": [],
    "advice": []
}

def load_or_train_model():
    """Load or train the machine learning model"""
    try:
        model = joblib.load('crop_model.pkl')
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != 9:
            raise ValueError("Model feature mismatch. Retraining...")
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Model loading failed: {str(e)}. Training new model...")
        return train_new_model()

def train_new_model():
    """Train a new Random Forest Classifier model"""
    data = []
    for crop, params in CROP_RANGES.items():
        for _ in range(100):  # Generate synthetic data points
            data.append({
                "N": np.random.uniform(*params["N"]),
                "P": np.random.uniform(*params["P"]),
                "K": np.random.uniform(*params["K"]),
                "EC": np.random.uniform(*params["EC"]),
                "pH": np.random.uniform(*params["pH"]),
                "lux": np.random.uniform(*params["lux"]),
                "temp": np.random.uniform(*params["temp"]),
                "humidity": np.random.uniform(*params["humidity"]),
                "moisture": np.random.uniform(*params["moisture"]),
                "crop": crop
            })
    
    df = pd.DataFrame(data)
    X = df[['N', 'P', 'K', 'EC', 'pH', 'lux', 'temp', 'humidity', 'moisture']]
    y = df['crop']
    
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'crop_model.pkl')
    return model

def analyze_conditions(data):
    """Analyze soil conditions against crop requirements"""
    recommendations = []
    problems = []
    for crop, params in CROP_RANGES.items():
        crop_ok = True
        for param, (low, high) in params.items():
            if param in data:
                value = data[param]
                if value < low or value > high:
                    crop_ok = False
                    problems.append(f"{param} {value:.1f} out of {crop} range ({low}-{high})")
        if crop_ok:
            recommendations.append(crop)
    return recommendations, problems

def generate_advice(data, problems):
    """Generate actionable advice based on soil conditions"""
    advice = []
    if data['EC'] > 2:
        advice.append("High salinity detected! Consider soil leaching")
    if data['pH'] < 5.5:
        advice.append("Soil too acidic! Add lime to raise pH")
    if data['pH'] > 7.5:
        advice.append("Soil too alkaline! Add sulfur to lower pH")
    if problems and not any("perfect" in p for p in problems):
        advice.append("Consider soil amendments or greenhouse adjustments")
    return advice

def update_analysis_results(data, recommendations, problems):
    """Update the global analysis results"""
    global latest_analysis
    latest_analysis = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "data": data,
        "recommendations": recommendations,
        "problems": problems,
        "advice": generate_advice(data, problems)
    }

def process_sensor_data(data):
    """Process incoming sensor data and update recommendations"""
    print("\n=== NEW ANALYSIS ===")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    clean_data = {
        'N': float(data.get('soilN', 0)),
        'P': float(data.get('soilP', 0)),
        'K': float(data.get('soilK', 0)),
        'EC': float(data.get('soilEC', 0)),
        'pH': float(data.get('soilPH', 7)),
        'lux': float(data.get('lux', 0)),
        'temp': float(data.get('airTemp', 25)),
        'humidity': float(data.get('humidity', 50)),
        'moisture': float(data.get('soilMoisture', 1.0))
    }

    # Print current sensor readings
    for k, v in clean_data.items():
        print(f"{k}: {v:.1f}")

    # Get crop recommendations
    model = load_or_train_model()
    input_data = pd.DataFrame([clean_data])
    probas = model.predict_proba(input_data)[0]
    crops = model.classes_
    top3 = sorted(zip(crops, probas), key=lambda x: -x[1])[:3]

    # Analyze conditions and update results
    recs, problems = analyze_conditions(clean_data)
    update_analysis_results(clean_data, top3, problems)

    # Print analysis results
    print("\nTOP CROP RECOMMENDATIONS:")
    for crop, prob in top3:
        print(f"- {crop}: {prob:.1%} match")

    if recs:
        print("\n✅ IDEAL CROPS:")
        for crop in recs:
            print(f"- {crop}")
    else:
        print("\n⚠️ NO PERFECT MATCHES FOUND")

    if problems:
        print("\nMAJOR ISSUES DETECTED:")
        for problem in set(problems):
            print(f"- {problem}")

    print("\nADVICE:")
    for advice_item in latest_analysis['advice']:
        print(f"- {advice_item}")



@app.route('/api/data')
def get_data():
    """API endpoint to get current analysis data"""
    return jsonify(latest_analysis)

@app.route('/healthz')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy", "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}), 200

# Firebase Listener
def setup_firebase_listener():
    """Set up Firebase realtime database listener"""
    ref = db.reference('/sensorData')
    
    def callback(event):
        if event.event_type == 'put':
            latest_data = db.reference('/sensorData').get()
            if isinstance(latest_data, dict):
                latest_key = sorted(latest_data.keys())[-1]
                data = latest_data[latest_key]
                if isinstance(data, dict):
                    process_sensor_data(data)

    ref.listen(callback)

# Application Startup
def run_flask_server():
    """Run the Flask web server"""
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    print("🚜 Starting Real-time Crop Recommendation System")
    print("🌐 Starting Flask server...")
    
    # Start Flask in a separate thread
    flask_thread = Thread(target=app.run, kwargs={
        'host': '0.0.0.0',
        'port': int(os.getenv('PORT', 5000)),
        'debug': False,
        'use_reloader': False
    })
    flask_thread.daemon = True
    flask_thread.start()
    
    print("🔁 Setting up Firebase listener...")
    try:
        setup_firebase_listener()
        print("✅ System ready and listening for updates...")
    except Exception as e:
        print(f"❌ Failed to setup Firebase listener: {e}")
    
    # Keep the main thread alive
    while True:
        time.sleep(1)
