from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
import os

# Buat instance FastAPI
app = FastAPI()

# Tambahkan Prometheus instrumentator
Instrumentator().instrument(app).expose(app)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "../model/car_price_model.pkl")
model = joblib.load(model_path)

# Struktur input data
class CarInput(BaseModel):
    horsepower: float
    curbweight: float
    enginesize: float

# Endpoint untuk prediksi harga
@app.post("/predict")
def predict_price(data: CarInput):
    input_data = [[data.horsepower, data.curbweight, data.enginesize]]
    prediction = model.predict(input_data)
    return {"predicted_price": round(prediction[0], 2)}

# Endpoint untuk menampilkan HTML frontend
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    # Path absolut ke index.html
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "../frontend/index.html")
    with open(html_path, "r") as f:
        return f.read()

