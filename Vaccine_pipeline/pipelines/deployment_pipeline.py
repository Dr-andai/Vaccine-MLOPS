import json

import os

import numpy as np
import pandas as pd


from zenml import pipeline, step
from zenml.config import DockerSettings

from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer)

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model


docker_settings = DockerSettings(required_integrations=[MLFLOW])

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.75

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    return accuracy >= config.min_accuracy

@pipeline(enable_cache=True, settings={"docker": docker_settings})

def continous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.75,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    acc = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(acc)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout= timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    ()