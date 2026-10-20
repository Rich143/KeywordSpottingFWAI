# MLflow tracking store

`TrainModel.ipynb` logs one MLflow run per training session (run name = `MODEL_VERSION`): the
hyperparameters and preprocessing settings, the dataset manifest (which clip went into which split),
per-epoch metrics, test metrics for the Keras/int8/float models, the models themselves and plots.
Its Board Eval section runs `snapshot_eval.py` on the int8 model and adds the board
recall / false-accept numbers to the same run, so "Run All" gives a complete run.

```
mlflow/
  mlflow.db     SQLite: experiments, runs, params, metrics, tags
  artifacts/    <run_id>/artifacts/{models,plots,reports,dataset}/...
```

Both are stored with **Git LFS** (see the repo's `.gitattributes`).

## Viewing

```
cd HostSW
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

Then open http://127.0.0.1:5000. Experiment `kws`; tick runs and click Compare for side-by-side
params/metrics.

## Committing (backup)

1. Make sure the notebook's `mlflow.end_run()` cell has run (Run All does it) and stop `mlflow ui`,
   so nothing is writing to `mlflow.db`.
2. Delete throwaway runs first if you don't want them in history: delete them in the UI (that only
   marks them deleted), then purge them for good, from `HostSW/`:
   ```
   mlflow gc --backend-store-uri sqlite:///mlflow/mlflow.db --tracking-uri sqlite:///mlflow/mlflow.db
   ```
   `gc` needs `--tracking-uri` as well, to find the runs' artifacts. It leaves each purged run's
   empty `artifacts/<run_id>/` folder behind, which git ignores.
3. Commit `HostSW/mlflow/` in the same commit as the new `tflite_models/kws_<ver>_*` files and the
   updated `eval_recordings/eval_history.csv`.

`mlflow.db` is binary, so LFS keeps a full copy of it per commit (around 1 MB; the artifacts add
about 1-2 MB per run). Git can't merge it either, so only commit runs from one branch (`main`).

## Restoring on another machine / path

MLflow stores the experiment's artifact location as an absolute `file://` path. A clone at the
same path (`/Users/gerald/SoftwareProjects/KeywordSpottingFWAI`) works as-is. Elsewhere, point it at
the new location once (from `HostSW/`):

```
sqlite3 mlflow/mlflow.db "UPDATE experiments SET artifact_location = 'file://$PWD/mlflow/artifacts' WHERE name = 'kws';
UPDATE runs SET artifact_uri = 'file://$PWD/mlflow/artifacts/' || run_uuid || '/artifacts';"
```
