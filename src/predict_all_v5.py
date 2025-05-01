import pandas as pd
import json
import mlflow
import numpy as np

# MLflow URI
mlflow.set_tracking_uri("http://127.0.0.1:5500")

# 1. Veriyi oku
df_orig = pd.read_csv("data/customer_churn.csv")
df = df_orig.copy()

# 2. Ön işleme
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

# 3. Doğru feature sırasını uygula
with open("model_input_columns5.json") as f:
    columns = json.load(f)

for col in columns:
    if col not in df.columns:
        df[col] = 0
df = df[columns]

# 4. Dummy sütunları bool yap (numerik hariç)
for col in df.columns:
    if df[col].dropna().isin([0, 1]).all() and col not in ["SeniorCitizen", "tenure"]:
        df[col] = df[col].astype(bool)

# 5. Modeli yükle (Version 5)
model = mlflow.pyfunc.load_model("models:/CustomerChurnModel/5")

# 6. Tahmin yap
predictions = model.predict(df)

# 7. Orijinal veriye tahminleri ekle
df_orig = df_orig.loc[df.index]
df_orig["prediction"] = predictions

# 8. Riskli müşterileri ayır
risky = df_orig[df_orig["prediction"] == 1]
print(f"🔍 (v5) Bırakacak müşteri sayısı: {len(risky)}")

# 9. Dosyaları kaydet
if not risky.empty:
    risky.to_csv("churn_riski_yuksekler_v5.csv", index=False)
    print("📁 churn_riski_yuksekler_v5.csv oluşturuldu.")
else:
    print("⚠️ Hiç churn riski yüksek müşteri bulunamadı.")

df_orig.to_csv("predictions_v5.csv", index=False)
print("📁 predictions_v5.csv oluşturuldu.")
