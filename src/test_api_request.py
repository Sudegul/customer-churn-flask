import pandas as pd
import json
import requests

# 📂 Veri yolun
df = pd.read_csv("data/customer_churn.csv")
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

# 💡 Feature listesi (logladığın .json dosyasından)
with open("model_input_columns13.json") as f:
    columns = json.load(f)

# Eksik sütunları tamamla ve sıralamayı eşleştir
for col in columns:
    if col not in df.columns:
        df[col] = 0
df = df[columns]

# Boolean dönüşüm (MLflow serve uyumu için)
for col in df.columns:
    if df[col].dropna().isin([0, 1]).all() and col not in ["SeniorCitizen", "tenure"]:
        df[col] = df[col].astype(bool)

# 🔢 İlk 5 satırı gönder
input_data = {"dataframe_records": df.iloc[:5].to_dict(orient="records")}

res = requests.post(
    url="http://127.0.0.1:5002/invocations",  # 🔁 doğru port!
    headers={"Content-Type": "application/json"},
    data=json.dumps(input_data)
)

try:
    print("✅ Tahmin:", res.json())
except Exception as e:
    print("❌ Hata:", str(e))
