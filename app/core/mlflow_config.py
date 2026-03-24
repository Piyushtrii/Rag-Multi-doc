import mlflow


def init_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("CRAG_Experiment")