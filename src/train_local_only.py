import pandas as pd
import mlflow
import mlflow.xgboost
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from mlflow.models.signature import infer_signature

# 🔧 MLflow ayarları
mlflow.set_tracking_uri("http://127.0.0.1:5501")
mlflow.set_experiment("Default")

print("📂 Veri yükleniyor...")
df = pd.read_csv("data/customer_churn.csv")

# 🧹 Temizleme
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 🔀 Split & SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

# 🔍 Model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.06,
    max_depth=3,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 🧪 MLflow loglama (metrics only)
with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # ✍️ Feature listesi kaydet
    with open("model_input_columns13.json", "w") as f:
        json.dump(X_test.columns.tolist(), f)

print("✅ Metrikler loglandı (MLflow UI'de görünür).")

# 💾 Modeli diske kaydet (REST API için kullanılacak)
mlflow.xgboost.save_model(
    xgb_model=model,
    path="exported_model_v13"
)

print("✅ Model klasörü oluşturuldu: exported_model_v13")
