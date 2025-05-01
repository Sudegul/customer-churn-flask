import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5500")

run_id = "7bf0fffec3ce43c6a9eaa28b89dde0c4"

result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="CustomerChurnModel"
)

print("✅ Model tekrar kaydedildi:")
print("Yeni versiyon:", result.version)
