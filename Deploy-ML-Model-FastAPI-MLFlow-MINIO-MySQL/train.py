import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __():


    import pandas as pd
    import numpy as np
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from urllib.parse import urlparse
    from mlflow.tracking import MlflowClient
    import mlflow.sklearn
    from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository


    return (
        MlflowClient,
        RandomForestRegressor,
        RunsArtifactRepository,
        mean_absolute_error,
        mean_squared_error,
        mlflow,
        np,
        os,
        pd,
        r2_score,
        train_test_split,
        urlparse,
    )


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
