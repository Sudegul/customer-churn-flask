import pandas as pd
import json
import mlflow
import numpy as np

# 🔹 1. MLflow bağlantısı (senin portuna göre ayarlı)
mlflow.set_tracking_uri("http://127.0.0.1:5500")

# 🔹 2. Veriyi yükle
df_orig = pd.read_csv("data/customer_churn.csv")
df = df_orig.copy()

# 🔹 3. Ön işleme
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

# 🔹 4. Feature sırasını yükle
with open("model_input_columns3.json") as f:
    columns = json.load(f)

for col in columns:
    if col not in df.columns:
        df[col] = 0
df = df[columns]

# 🔹 5. Sadece dummy'leri bool yap (numerik bozulmasın)
for col in df.columns:
    if df[col].dropna().isin([0, 1]).all() and col not in ["SeniorCitizen", "tenure"]:
        df[col] = df[col].astype(bool)

# 🔹 6. Version 3 modelini yükle
model = mlflow.pyfunc.load_model("models:/CustomerChurnModel/3")

# 🔹 7. Tahmin yap
predictions = model.predict(df)

# 🔬 Dağılımı göster
unique, counts = np.unique(predictions, return_counts=True)
print("🔬 Version 3 Tahmin dağılımı:", dict(zip(unique, counts)))

# 🔹 8. Tahminleri orijinal veriyle birleştir
df_orig = df_orig.loc[df.index]
df_orig["prediction"] = predictions

# 🔹 9. Churn riski yüksek olanları al
risky = df_orig[df_orig["prediction"] == 1]
print(f"🔍 (v3) Bırakacak müşteri sayısı: {len(risky)}")

# 🔹 10. Sonuçları kaydet
if not risky.empty:
    risky.to_csv("churn_riski_yuksekler_v3.csv", index=False)
    print("📁 churn_riski_yuksekler_v3.csv başarıyla oluşturuldu.")
else:
    print("⚠️ churn_riski_yuksekler_v3.csv için bırakacak müşteri bulunamadı.")

# (Opsiyonel) Tüm tahminleri de yedekle
df_orig.to_csv("predictions_v3.csv", index=False)
print("📁 predictions_v3.csv başarıyla yazıldı.")
