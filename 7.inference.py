import os
import mlflow.pyfunc
from flask import Flask, request, jsonify
from prometheus_client import generate_latest, Counter, Histogram, Gauge
import time
import pandas as pd
# Removed from pathlib import Path as it's no longer strictly needed with hardcoded path

app = Flask(__name__)

# --- Prometheus Metrics ---
PREDICTIONS_TOTAL = Counter(
    'ml_model_predictions_total',
    'Total number of predictions made by the ML model.'
)

PREDICTION_DURATION_SECONDS = Histogram(
    'ml_model_prediction_duration_seconds',
    'Histogram of prediction duration in seconds.',
    buckets=(.001, .005, .01, .025, .05, .075, .1, .25, .5, 1.0, 2.5, 5.0, 10.0, float('inf'))
)

MODEL_LOAD_SUCCESS = Gauge(
    'ml_model_load_success',
    'Gauge indicating if the ML model was loaded successfully (1 for success, 0 for failure).'
)

# --- Model Loading Logic ---
# !!! IMPORTANT: This path is for LOCAL (non-Docker) execution. !!!
# You MUST have created the E:\mlmodel\repacked_tuned_model folder.
MODEL_PATH = "file:///E:/mlmodel/repacked_tuned_model" # <--- Hardcoded SHORT PATH for local run

# Fallback to environment variable (e.g., if Docker container sets it)
# The Docker container will use the ENV var from docker-compose.yaml
MODEL_PATH = os.getenv('MODEL_PATH', MODEL_PATH)

model = None

try:
    print(f"Attempting to load model from: {MODEL_PATH}")
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    MODEL_LOAD_SUCCESS.set(1)
    print("Model loaded successfully!")
except Exception as e:
    MODEL_LOAD_SUCCESS.set(0)
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 503

    start_time = time.time()
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400

        input_df = pd.DataFrame(data)

        predictions = model.predict(input_df)
        PREDICTIONS_TOTAL.inc()

        end_time = time.time()
        PREDICTION_DURATION_SECONDS.observe(end_time - start_time)

        return jsonify(predictions.tolist())
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    return generate_latest(), 200

@app.route('/health')
def health_check():
    if model is not None:
        return "Model is loaded and ready", 200
    return "Model not loaded", 503

if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 5001)) # Running on 5001
    app.run(host=host, port=port, debug=False)