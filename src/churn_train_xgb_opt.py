import pandas as pd
import json
import mlflow
import mlflow.xgboost
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# MLflow bağlantısı
mlflow.set_tracking_uri("http://127.0.0.1:5500")
mlflow.set_experiment("Default")

# 1. Veri yükle
df = pd.read_csv("data/customer_churn.csv")

# 2. Ön işleme
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

# 3. Feature listesi kaydet
with open("model_input_columns8.json", "w") as f:
    json.dump(df.drop("Churn", axis=1).columns.tolist(), f)

# 4. Train/test ayır
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. SMOTE uygulaması
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 6. Sınıf dengesine göre scale_pos_weight hesapla
neg, pos = y_train_resampled.value_counts()
scale_pos_weight = neg / pos

# 7. XGBoost modeli (optimize parametrelerle)
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)

# 8. MLflow loglama
with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    signature = infer_signature(X_test, y_pred)
    input_example = X_test.iloc[:1]

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
        registered_model_name="CustomerChurnModel"
    )

    print(f"✅ Optimize model loglandı → Version 8. Run ID: {run.info.run_id}")
