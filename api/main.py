from fastapi import FastAPI, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
import os

# ✅ Buat instance FastAPI lebih dulu
app = FastAPI()

# ✅ Tambahkan Prometheus instrumentator
Instrumentator().instrument(app).expose(app)

# ✅ Load model
model_path = os.path.join(os.path.dirname(__file__), "../model/car_price_model.p                                                                             kl")
model = joblib.load(model_path)

# ✅ Struktur input data untuk API
class CarInput(BaseModel):
    horsepower: float
    curbweight: float
    enginesize: float

# ✅ Endpoint API POST /predict
@app.post("/predict")
def predict_price(data: CarInput):
    input_data = [[data.horsepower, data.curbweight, data.enginesize]]
    prediction = model.predict(input_data)
    return {"predicted_price": round(prediction[0], 2)}

# ✅ Endpoint HTML (Form input dari browser)
@app.get("/predict", response_class=HTMLResponse)
def form_predict():
    return """
    <html>
        <body>
            <h2>Prediksi Harga Mobil</h2>
            <form action="/predict" method="post">
                Horsepower: <input type="number" name="horsepower"><br>
                Curbweight: <input type="number" name="curbweight"><br>
                Enginesize: <input type="number" name="enginesize"><br>
                <input type="submit">
            </form>
        </body>
    </html>
    """

# ✅ Endpoint form POST handler
@app.post("/predict", response_class=HTMLResponse)
def form_predict_post(horsepower: float = Form(...), curbweight: float = Form(..                                                                             .), enginesize: float = Form(...)):
    input_data = [[horsepower, curbweight, enginesize]]
    prediction = model.predict(input_data)
    return f"<h3>Harga mobil diprediksi: {round(prediction[0], 2)}</h3>"

# ✅ Endpoint untuk frontend bawaan
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "../frontend/index.html")
    with open(html_path, "r") as f:
        return f.read()

