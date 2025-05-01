import pandas as pd
import json
import mlflow
import numpy as np

# 🔹 1. MLflow Tracking Server URI (5500 ya da senin portun hangisiyse)
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

# 🔹 4. Eğitimde kullanılan feature sırasını yükle
with open("model_input_columns3.json") as f:
    columns = json.load(f)

# 🔹 5. Eksik sütunları tamamla ve sıralamayı eşitle
for col in columns:
    if col not in df.columns:
        df[col] = 0
df = df[columns]

# 🔹 6. Bozulabilecek numerik sütunları bozma, sadece dummy'leri bool yap
for col in df.columns:
    if df[col].dropna().isin([0, 1]).all() and col not in ["SeniorCitizen", "tenure"]:
        df[col] = df[col].astype(bool)

# 🔹 7. Modeli yükle (Model Registry'den)
model = mlflow.pyfunc.load_model("models:/CustomerChurnModel/1")

# 🔹 8. Tahmin yap
predictions = model.predict(df)

# 🔬 Tahmin dağılımını göster
unique, counts = np.unique(predictions, return_counts=True)
print("🔬 Tahmin dağılımı:", dict(zip(unique, counts)))

# 🔹 9. Orijinal veriye tahminleri ekle
df_orig = df_orig.loc[df.index]
df_orig["prediction"] = predictions

# 🔹 10. Riskli müşterileri ayır
risky = df_orig[df_orig["prediction"] == 1]
print(f"🔍 Bırakacak müşteri sayısı: {len(risky)}")

# 🔹 11. CSV olarak kaydet (boşsa bile uyarı ver)
if not risky.empty:
    risky.to_csv("churn_riski_yuksekler2.csv", index=False)
    print("📁 churn_riski_yuksekler2.csv başarıyla oluşturuldu.")
else:
    print("⚠️ Hiç bırakma riski taşıyan müşteri bulunamadı. CSV dosyası yazılmadı.")

# (İsteğe bağlı) Tüm tahminleri de ayrı dosyaya yaz
df_orig.to_csv("predictions.csv", index=False)
print("📁 Tüm tahminler predictions.csv dosyasına yazıldı.")
