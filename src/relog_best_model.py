import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiment_name = "Serve Local Model 2"
client = MlflowClient()

try:
    experiment_id = client.create_experiment(name=experiment_name)
    print("✅ Yeni deney oluşturuldu:", experiment_id)
except Exception as e:
    print("ℹ️ Deney zaten mevcut:", e)

mlflow.set_experiment(experiment_name)

df = pd.read_csv("data/customer_churn.csv")
df = df.dropna()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=80, max_depth=17, random_state=42)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))

with mlflow.start_run() as run:
    print("🟢 Run ID:", run.info.run_id)
    mlflow.log_param("n_estimators", 80)
    mlflow.log_param("max_depth", 17)
    mlflow.log_metric("accuracy", acc)

    try:
        print("📦 Model loglama başlıyor...")
        mlflow.sklearn.log_model(clf, "model")
        print("✅ Model başarıyla loglandı.")
    except Exception as e:
        print("❌ Model loglama HATASI:", e)

import json
with open("model_input_columns.json", "w") as f:
    json.dump(X_train.columns.tolist(), f)
