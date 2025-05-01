import pandas as pd
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Churn Hyperopt")

# Veri yükle ve ön işleme
df = pd.read_csv("data/customer_churn.csv")
df = df.dropna()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hedef: maximize accuracy
def objective(params):
    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            random_state=42
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        return {"loss": -acc, "status": STATUS_OK}

# Arama alanı
search_space = {
    "n_estimators": hp.quniform("n_estimators", 50, 300, 10),
    "max_depth": hp.quniform("max_depth", 2, 20, 1)
}

# Denemeleri başlat
with mlflow.start_run(run_name="Hyperopt Tuning"):
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )
    mlflow.log_params(best_result)
