"""
MLflow setup shared by TrainModel.ipynb and the eval scripts, so they all log to the same
store (HostSW/mlflow/, see its README) and experiment.

Run types (the `run_type` tag):
  training  a TrainModel.ipynb run, named after the model version (e.g. "v10"). Evals run
            from the notebook log into it.
  eval      a standalone snapshot_eval.py run, named "<version> <dataset>-<split> eval". It's
            nested under that version's training run (mlflow.parentRunId) when there is one.
Training runs made before the tag existed have no run_type; they count as training.

UI: cd HostSW && mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
"""

import os
import pathlib

import mlflow

HOSTSW_DIR = pathlib.Path(__file__).resolve().parent.parent
MLFLOW_DIR = HOSTSW_DIR / "mlflow"
# Absolute, so the scripts find the store whatever directory they're run from
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_DIR = MLFLOW_DIR / "artifacts"
MLFLOW_EXPERIMENT = "kws"

RUN_TYPE_TAG = "run_type"
RUN_TYPE_TRAINING = "training"
RUN_TYPE_EVAL = "eval"


def init_mlflow():
    """Points MLflow at HostSW/mlflow and selects the experiment (creating it, with its
    artifacts in mlflow/artifacts, if needed). Returns the experiment id."""
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        # Without an explicit location MLflow puts artifacts in ./mlruns
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri())
    return mlflow.set_experiment(MLFLOW_EXPERIMENT).experiment_id


def find_training_run(model_version):
    """The latest training run named model_version, or None."""
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        filter_string=f"tags.mlflow.runName = '{model_version}'",
        order_by=["start_time DESC"],
        output_format="list",
    )
    for run in runs:
        if run.data.tags.get(RUN_TYPE_TAG, RUN_TYPE_TRAINING) == RUN_TYPE_TRAINING:
            return run
    return None


def start_eval_run(model_version, dataset, split, params):
    """Creates a standalone eval run (not made active, so it can't capture other logging)
    and returns its id. Nested under the version's training run if there is one."""
    experiment_id = init_mlflow()
    tags = {
        RUN_TYPE_TAG: RUN_TYPE_EVAL,
        "model_version": model_version,
        "eval_dataset": dataset,
        "eval_split": split,
    }
    parent = find_training_run(model_version)
    if parent is not None:
        tags["mlflow.parentRunId"] = parent.info.run_id
    client = mlflow.MlflowClient()
    run = client.create_run(experiment_id, run_name=f"{model_version} {dataset}-{split} eval", tags=tags)
    for key, value in params.items():
        client.log_param(run.info.run_id, key, value)
    return run.info.run_id
