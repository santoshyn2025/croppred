from flask import Flask, jsonify
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

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],  # Allow all origins for API endpoints
        "methods": ["GET"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize Firebase
def initialize_firebase():
    try:
        firebase_creds = json.loads(os.environ['FIREBASE_CREDENTIALS'])
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-agri-2d3da-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
        print("✅ Firebase initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Firebase: {str(e)}")
        raise

try:
    initialize_firebase()
except Exception as e:
    print(f"Failed to initialize Firebase: {e}")

# Crop Ranges Configuration
CROP_RANGES = {
    "rice": {"N": (20, 50), "P": (30, 60), "K": (5, 15), "EC": (25, 30), "pH": (5.5, 7.0),
             "moisture": (1.0, 2.5), "lux": (5000, 10000), "temp": (25, 35), "humidity": (60, 85)},
    # ... (include all other crops with their ranges)
}

# Global analysis results
latest_analysis = {
    "timestamp": "",
    "data": {},
    "recommendations": [],
    "problems": [],
    "advice": []
}

def load_or_train_model():
    try:
        model = joblib.load('crop_model.pkl')
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != 9:
            raise ValueError("Model feature mismatch")
        print("Model loaded successfully")
        return model
    except Exception:
        print("Training new model...")
        return train_new_model()

def train_new_model():
    data = []
    for crop, params in CROP_RANGES.items():
        for _ in range(100):
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
    advice = []
    if data['EC'] > 2:
        advice.append("High salinity detected! Consider soil leaching")
    if data['pH'] < 5.5:
        advice.append("Soil too acidic! Add lime to raise pH")
    if data['pH'] > 7.5:
        advice.append("Soil too alkaline! Add sulfur to lower pH")
    if problems:
        advice.append("Consider soil amendments or greenhouse adjustments")
    return advice

def update_analysis_results(data, recommendations, problems):
    global latest_analysis
    latest_analysis = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "data": data,
        "recommendations": recommendations,
        "problems": problems,
        "advice": generate_advice(data, problems)
    }

def process_sensor_data(data):
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

    model = load_or_train_model()
    input_data = pd.DataFrame([clean_data])
    probas = model.predict_proba(input_data)[0]
    crops = model.classes_
    top3 = sorted(zip(crops, probas), key=lambda x: -x[1])[:3]

    recs, problems = analyze_conditions(clean_data)
    update_analysis_results(clean_data, top3, problems)

# API Endpoints
@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "CropPred API",
        "endpoints": {
            "data": "/api/data",
            "health": "/healthz"
        }
    })

@app.route('/api/data')
def get_data():
    return jsonify(latest_analysis)

@app.route('/healthz')
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}), 200

# Firebase Listener
def setup_firebase_listener():
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

# Main Application
if __name__ == "__main__":
    print("🚜 Starting Real-time Crop Recommendation System")
    
    # Start Flask server
    port = int(os.getenv("PORT", 5000))
    flask_thread = Thread(target=app.run, kwargs={
        'host': '0.0.0.0',
        'port': port,
        'debug': False,
        'use_reloader': False
    })
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start Firebase listener
    try:
        setup_firebase_listener()
        print("✅ System ready and listening for updates...")
    except Exception as e:
        print(f"❌ Failed to setup Firebase listener: {e}")
    
    # Keep main thread alive
    while True:
        time.sleep(1)
