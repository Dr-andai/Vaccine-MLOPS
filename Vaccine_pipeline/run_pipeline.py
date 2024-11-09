from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
import mlflow

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/Users/Hp/OneDrive/OLD FILES/Documents/PROJECTS/Vaccine Decision Trees/Vaccine_pipeline/data/Covid Vaccine hesitancy.csv")

# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# # Create a new MLflow Experiment
# mlflow.set_experiment("MLflow Quickstart")

# mlflow ui --backend-store-uri "file:C:/Users/Hp/AppData/Roaming/zenml/local_stores/42a53655-a67a-4798-a660-b712a5d4d12d/mlruns"