import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veri seti
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

# MLflow deney başlat
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, max_depth=3)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", acc)

    # Modeli logla
    mlflow.sklearn.log_model(clf, "model")

