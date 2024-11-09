from typing import cast

import click

from pipelines.deployment_pipeline import (
    continous_deployment_pipeline, 
    inference_pipeline
    )

from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer)

from zenml.integrations.mlflow.services import MLFlowDeploymentService



DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default = DEPLOY_AND_PREDICT,
    help = "All will be run",
)
@click.option(
    "--min-accuracy",
    default = 0.75,
    help = "Minimu accuracy required to deploy the model",
)

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        continous_deployment_pipeline(
            min_accuracy= min_accuracy,
            workers = 3,
            timeouts = 60,
            )
        
    # if predict:
    #     inference_pipeline(
    #         pipeline_name ="continous_deployment_pipeline",
    #         pipeline_step_name="mlflow_model_deployer_step",
    #     )

    print(
        "You can run:\n"
        f"mlfow ui --backend-store-uri '{get_tracking_uri()}"
    )

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLFlowprediction server is running locally"
                f" {service.prediction_url }\n"
            )
        elif service.is_failed:
            print(
                f"The ML flow prediction server is a failed state:\n"
                f" Last state: '{service.status.state.value}'"
            )
    else:
        print(
            "No MLFlow server is running"
        )


if __name__ == "__main__":
    run_deployment()
