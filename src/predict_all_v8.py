import pandas as pd
import json
import mlflow
import numpy as np

# MLflow ayarı
mlflow.set_tracking_uri("http://127.0.0.1:5500")

# 🔹 Veriyi yükle
df_orig = pd.read_csv("data/customer_churn.csv")
df = df_orig.copy()

# 🔹 Ön işleme
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

# 🔹 Modelin beklediği sütun sıralaması
with open("model_input_columns8.json") as f:
    columns = json.load(f)

# Eksik sütunları sıfırla
for col in columns:
    if col not in df.columns:
        df[col] = 0
df = df[columns]

# 🔹 Boolean dönüştürme (MLflow schema hatasından korunmak için)
for col in df.columns:
    if df[col].dropna().isin([0, 1]).all() and col not in ["SeniorCitizen", "tenure"]:
        df[col] = df[col].astype(bool)

# 🔹 Modeli yükle
model = mlflow.pyfunc.load_model("models:/CustomerChurnModel/8")

# 🔹 Tahmin yap
predictions = model.predict(df)
df_orig = df_orig.loc[df.index]
df_orig["prediction"] = predictions

# 🔹 Churn riski yüksek olanları al
risky = df_orig[df_orig["prediction"] == 1]
print(f"🔍 Version 8 tahmin sonucu: {len(risky)} müşteri bırakacak gibi görünüyor.")

# 🔹 Dosya kaydet
df_orig.to_csv("predictions_v8.csv", index=False)
print("📁 predictions_v8.csv dosyası oluşturuldu.")

if not risky.empty:
    risky.to_csv("churn_riski_yuksekler_v8.csv", index=False)
    print("📁 churn_riski_yuksekler_v8.csv dosyası oluşturuldu.")
else:
    print("⚠️ Hiç churn riski yüksek müşteri bulunamadı.")
