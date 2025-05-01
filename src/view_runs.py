import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment = mlflow.get_experiment_by_name("Churn Hyperopt")

client = mlflow.tracking.MlflowClient()

runs = client.search_runs(experiment_ids=[experiment.experiment_id])

for run in runs:
    run_id = run.info.run_id
    acc = run.data.metrics.get("accuracy", "N/A")
    params = run.data.params
    print(f"Run ID: {run_id}, Accuracy: {acc}, Params: {params}")


                                       ### SONUÇLAR


'''
Run ID: 5d1dd0098fca4f22b540a84d648e0104, Accuracy: 0.7544357700496807, Params: {'max_depth': '16.0', 'n_estimators': '100.0'}
Run ID: 06b0ccebe996407ea505fceb62de2d8d, Accuracy: 0.7352732434350603, Params: {'max_depth': '10.0', 'n_estimators': '100.0'}
Run ID: 5c1a75bd5d614e3fb23e99cc88d116c3, Accuracy: 0.7352732434350603, Params: {'max_depth': '13.0', 'n_estimators': '110.0'}
Run ID: 2af06360d4174b98bebcfddf15f3e5fa, Accuracy: 0.7352732434350603, Params: {'max_depth': '5.0', 'n_estimators': '160.0'}
Run ID: 77798273805040c4aa3d218ba0b272f6, Accuracy: 0.7352732434350603, Params: {'max_depth': '2.0', 'n_estimators': '200.0'}
Run ID: 7f9d43fcedab405b8b51ad94cdf1a8ba, Accuracy: 0.7452093683463449, Params: {'max_depth': '16.0', 'n_estimators': '170.0'}
Run ID: 0e557ae4c5ec46c08f10d42c1211ecca, Accuracy: 0.7409510290986515, Params: {'max_depth': '16.0', 'n_estimators': '280.0'}
Run ID: f94ce8589c1c47e99b7d973f714f5afb, Accuracy: 0.7352732434350603, Params: {'max_depth': '7.0', 'n_estimators': '220.0'}
Run ID: 5fa3f6e6ee5d4d2ea0b2f71361d45f08, Accuracy: 0.7352732434350603, Params: {'max_depth': '8.0', 'n_estimators': '260.0'}
Run ID: 912a686cb01d4b8b96c4c02941763689, Accuracy: 0.7352732434350603, Params: {'max_depth': '12.0', 'n_estimators': '80.0'}
Run ID: 9f0f6ee734e04b4a8c48a18791671a64, Accuracy: 0.7352732434350603, Params: {'max_depth': '5.0', 'n_estimators': '220.0'}
Run ID: b18016c48c134ccba5040eb3e652c351, Accuracy: 0.7686302342086586, Params: {'max_depth': '17.0', 'n_estimators': '80.0'}
Run ID: 7f6fe62ef401415798a55495a5d10a2c, Accuracy: 0.7352732434350603, Params: {'max_depth': '5.0', 'n_estimators': '290.0'}
Run ID: 1db09c1122e9496da05142c9bc946761, Accuracy: 0.7374024130589071, Params: {'max_depth': '14.0', 'n_estimators': '300.0'}
Run ID: d4ccf4a35680493b81c0890b6c80ae4a, Accuracy: 0.7352732434350603, Params: {'max_depth': '2.0', 'n_estimators': '200.0'}
Run ID: b907791b7b2940da9b7fcd346ae6260f, Accuracy: 0.7352732434350603, Params: {'max_depth': '8.0', 'n_estimators': '180.0'}
Run ID: 860c716d8cee4cb885cf22abae711d9f, Accuracy: 0.7352732434350603, Params: {'max_depth': '4.0', 'n_estimators': '140.0'}
Run ID: 49700bf158b84d80a297c11db60af41f, Accuracy: 0.7352732434350603, Params: {'max_depth': '3.0', 'n_estimators': '150.0'}
Run ID: 6608890329f942c085bd0a266b33e31b, Accuracy: 0.7352732434350603, Params: {'max_depth': '5.0', 'n_estimators': '90.0'}
Run ID: 81656465376d49129a16acbada76d192, Accuracy: 0.7352732434350603, Params: {'max_depth': '8.0', 'n_estimators': '250.0'}
'''
