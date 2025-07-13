import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Baca data
df = pd.read_csv('../data/car_price.csv')

# 2. Cek kolom yang tersedia
print("Kolom yang tersedia:", df.columns.tolist())

# 3. Pilih fitur dan target yang sesuai
# Kita pakai fitur numerik yang sudah bersih
df = df[['horsepower', 'curbweight', 'enginesize', 'price']].dropna()

# 4. Siapkan X (fitur) dan y (target)
X = df[['horsepower', 'curbweight', 'enginesize']]
y = df['price']

# 5. Split data jadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Latih model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"📊 Mean Squared Error: {mse:.2f}")

# 8. Simpan model
joblib.dump(model, '../model/car_price_model.pkl')
print("✅ Model berhasil disimpan ke model/car_price_model.pkl")

