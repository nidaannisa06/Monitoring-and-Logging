import os
import time
import mlflow.pyfunc
from prometheus_client import start_http_server, Gauge
from mlflow.exceptions import MlflowException
# Removed from pathlib import Path as it's no longer strictly needed with hardcoded path

# --- Prometheus Metrics ---
HOUSE_PRICE_PREDICTION_GAUGE = Gauge(
    'ml_model_current_house_price_prediction',
    'Predicted house price from the ML model.'
)

PREDICTION_LATENCY_MS_GAUGE = Gauge(
    'ml_model_prediction_latency_ms',
    'Latency of house price prediction in milliseconds.'
)

MODEL_EXPORTER_LOAD_SUCCESS = Gauge(
    'ml_exporter_model_load_success',
    'Gauge indicating if the ML model was loaded successfully by the exporter (1 for success, 0 for failure).'
)

# --- Model Loading (for the exporter itself) ---
# !!! IMPORTANT: This path is for LOCAL (non-Docker) execution. !!!
# You MUST have created the E:\mlmodel\repacked_tuned_model folder.
MODEL_PATH = "file:///E:/mlmodel/repacked_tuned_model" # <--- Hardcoded SHORT PATH for local run

# Fallback to environment variable (e.g., if Docker container sets it)
# The Docker container will use the ENV var from docker-compose.yaml
MODEL_PATH = os.getenv('MODEL_EXPORTER_MODEL_PATH', MODEL_PATH)

model_exporter = None

def load_model_for_exporter():
    global model_exporter
    try:
        print(f"Exporter attempting to load model from: {MODEL_PATH}")
        model_exporter = mlflow.pyfunc.load_model(MODEL_PATH)
        MODEL_EXPORTER_LOAD_SUCCESS.set(1)
        print("Exporter model loaded successfully!")
    except MlflowException as e:
        MODEL_EXPORTER_LOAD_SUCCESS.set(0)
        print(f"MLflow specific error loading model for exporter: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        MODEL_EXPORTER_LOAD_SUCCESS.set(0)
        print(f"General error loading model for exporter: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus exporter started on port 8000")

    load_model_for_exporter()

    while True:
        time.sleep(5)