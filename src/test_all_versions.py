import pandas as pd
import json
import mlflow
import numpy as np
from mlflow.exceptions import MlflowException

# 🔹 MLflow tracking server portunu yaz (gerekirse değiştir)
mlflow.set_tracking_uri("http://127.0.0.1:5500")

# 🔹 Model adı
model_name = "CustomerChurnModel"

# 🔹 Veriyi yükle
df_orig = pd.read_csv("data/customer_churn.csv")
df = df_orig.copy()

# 🔹 Ön işleme
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

# 🔹 Feature sırasını oku
with open("model_input_columns3.json") as f:
    columns = json.load(f)

# 🔹 Eksik sütunları tamamla, sıraya koy
for col in columns:
    if col not in df.columns:
        df[col] = 0
df = df[columns]

# 🔹 Dummy sütunları bool yap (sayısallar hariç)
for col in df.columns:
    if df[col].dropna().isin([0, 1]).all() and col not in ["SeniorCitizen", "tenure"]:
        df[col] = df[col].astype(bool)

# 🔹 Test edilecek versiyon aralığı (1'den 15'e kadar dener)
for version in range(1, 16):
    try:
        print(f"\n🔄 Testing model version {version}...")
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
        predictions = model.predict(df)
        churn_count = np.sum(predictions == 1)
        print(f"✅ Version {version} → Bırakacak müşteri sayısı: {churn_count}")
    except MlflowException as e:
        print(f"⛔ Version {version} yüklenemedi: {e.error_code}")
