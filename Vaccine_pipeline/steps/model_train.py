import logging
import pandas as pd
from zenml import step

import mlflow
from src.model_dev import RandomForestClassifierModel
from sklearn.base import BaseEstimator, ClassifierMixin
from.config import ModelNameConfig


from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config:ModelNameConfig,
)-> ClassifierMixin:
    
    
    model = None

    if config.model_name=="RandomForest":
        mlflow.sklearn.autolog()
        model = RandomForestClassifierModel()
        trained_model = model.train(X_train, y_train)
        return trained_model
    else:
        raise ValueError("Model {} not supported".format(config.model_name))