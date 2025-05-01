import requests
import json

# Modelin beklediği tüm feature'ları al
with open("model_input_columns.json") as f:
    columns = json.load(f)

# Her şeyi sıfırla
sample_record = dict(zip(columns, [0] * len(columns)))

# Sadece bazıları 1 yapılır (örnek bir müşteri oluşturulur)
sample_record["tenure"] = 12
sample_record["MonthlyCharges"] = 75.30
sample_record["TotalCharges"] = 903.6
sample_record["gender_Male"] = 1
sample_record["Partner_Yes"] = 1
sample_record["Contract_One year"] = 1
sample_record["PaymentMethod_Electronic check"] = 1

# API için istek verisi
sample_input = {"dataframe_records": [sample_record]}

# Tahmin isteği gönder
res = requests.post(
    url="http://127.0.0.1:5002/invocations",  # 👈 5002 oldu
    headers={"Content-Type": "application/json"},
    data=json.dumps(sample_input)
)


print("📊 Tahmin sonucu:", res.json())
