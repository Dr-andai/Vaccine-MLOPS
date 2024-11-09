import logging
import pandas as pd
import mlflow
from zenml import step
from typing import Tuple

from src.evaluation_dev import accuracy
from sklearn.base import ClassifierMixin

from typing_extensions import Annotated
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   )-> Annotated[float, "acc"]:
    
    try:
        prediction = model.predict(X_test)
        accuracy_class = accuracy()
        acc = accuracy_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("acc", acc)
        return acc
    except Exception as e:
        logging.error("Error in evaluating model:{}".format(e))
        raise e