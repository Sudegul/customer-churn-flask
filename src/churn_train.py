import pandas as pd
import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# 🔹 MLflow ayarları
mlflow.set_tracking_uri("http://127.0.0.1:5500")
mlflow.set_experiment("Default")

# 🔹 1. Veri yükle
df = pd.read_csv("data/customer_churn.csv")

# 🔹 2. Temizlik
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 🔹 3. One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# 🔹 4. model_input_columns4.json dosyasını oluştur
with open("model_input_columns4.json", "w") as f:
    json.dump(df.drop("Churn", axis=1).columns.tolist(), f)

# 🔹 5. Eğitim/test ayrımı
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 6. Model eğitimi
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 🔹 7. MLflow loglama
with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Signature ve input example logla
    signature = infer_signature(X_test, y_pred)
    input_example = X_test.iloc[:1]

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
        registered_model_name="CustomerChurnModel"
    )

    print(f"✅ Model loglandı. Run ID: {run.info.run_id}")
