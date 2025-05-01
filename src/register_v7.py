import mlflow
import mlflow.sklearn
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier

# 🔹 Tracking URI ve experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Default")

# 🔹 Veriyi oku ve ön işle
df = pd.read_csv("data/customer_churn.csv")
df.drop("customerID", axis=1, inplace=True)
df = df.dropna()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 🔹 Input signature ve örnek kayıt oluştur
from mlflow.models.signature import infer_signature

input_example = X.iloc[[0]].to_dict(orient="records")[0]
signature = infer_signature(X, y)

# 🔹 MLflow Run
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
    clf.fit(X, y)

    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 5,
        "class_weight": "balanced"
    })

    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="ChurnPredictionModel"
    )
