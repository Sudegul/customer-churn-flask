import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

run_id = "b18016c48c134ccba5040eb3e652c351"
model_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model",
    dst_path="./downloaded_model"
)

print("Model indirildi →", model_path)
