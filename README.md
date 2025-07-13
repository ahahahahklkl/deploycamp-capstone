
## ⚙️ Instalasi & Jalankan (Local / Cloud)

1. **Clone repo**  
   `git clone https://github.com/ahahahahklkl/deploycamp-capstone.git && cd deploycamp-capstone`

2. **Pastikan Docker & Docker Compose terinstall**  

3. **Jalankan Docker Compose**  
   `docker-compose up -d`

4. **Akses aplikasi**  
   - API: `http://localhost:8080/predict`
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000` (default login: `admin/admin`)

## 🧠 Cara Kerja

### 1. Model Training  
- Notebook `notebook/train_model.py` membaca `data/car_price.csv`, melatih model (fitur: horsepower, curbweight, enginesize), meng-output MSE, dan menyimpan file model ke `model/car_price_model.pkl`.

### 2. FastAPI  
- Eksekusi `api/main.py` akan:
  - Load model ML
  - Expose endpoint `/predict` (POST)– menerima JSON input `{ "horsepower": float, "curbweight": float, "enginesize": float }`
  - Merespon `[predicted_price: float]`
  - Expose metrics via `prometheus_fastapi_instrumentator`

### 3. Monitoring  
- `monitoring/prometheus.yml`: scrape metrics endpoint `/metrics` di port 8000  
- Grafana sudah di-enable untuk tampilkan grafik metrics FastAPI

### 4. Deployment Cloud  
- Dapat dijalankan di VPS (contoh: Biznet Gio) dengan `$ docker-compose up -d`
- Volume `./model` dan `./monitoring/prom-data` dipetakan untuk persistensi

## 🎯 Test  
Contoh test menggunakan curl:
```bash
curl -X POST http://<ip-vm>:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"horsepower": 120, "curbweight": 1500, "enginesize": 130}'
