import os
import numpy as np
from math import exp
from sklearn import datasets

# Preprocess, get info of data
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

# model
import xgboost as xgb

# evaluation 
from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score
)

# tracking exp
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.client import MlflowClient

# hyper parameter optimization
from hyperopt import (
    fmin,
    hp,
    tpe,
    rand,
    SparkTrials,
    Trials,
    STATUS_OK
)
from hyperopt.pyll.base import scope

RANDOM_SEED = 0

mlflow.set_tracking_uri('http://0.0.0.0:4999')

data = datasets.load_breast_cancer(as_frame=True)
data_df = data.data
data_df['target'] = data.target
print(data_df)

if not os.path.isfile('data_profile.html'):
    data_profile = ProfileReport(data_df)
    data_profile.to_file(f'data_profile.html')

# Split data into train/validation and hold out test set
train_val, test = train_test_split(
    data_df, 
    test_size=0.15,
    shuffle=True, 
    random_state=RANDOM_SEED
)

X_train_val = train_val.drop(columns='target')
y_train_val = train_val.target

X_test = test.drop(columns='target')
y_test = test.target

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, 
    y_train_val,
    stratify=y_train_val,
    shuffle=True, 
    random_state=RANDOM_SEED
)

# Preprocessing data
power = PowerTransformer(method='yeo-johnson', standardize=True)
X_train = power.fit_transform(X_train)
X_val =power.transform(X_val)
X_test = power.transform(X_test)


# Setting search space for xgboost model
search_space = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
    'subsample': hp.uniform('subsample', .5, 1.0),
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 7),
    'reg_alpha': hp.loguniform('reg_alpha', -10, 10),
    'reg_lambda': hp.loguniform('reg_lambda', -10, 10),
    'gamma': hp.loguniform('gamma', -10, 10),
    'use_label_encoder': False,
    'verbosity': 0,
    'random_state': RANDOM_SEED
}

try:
    EXPERIMENT_ID = mlflow.create_experiment('xgboost-hyperopt')
except:
    EXPERIMENT_ID = dict(mlflow.get_experiment_by_name('xgboost-hyperopt'))

def train_model(params):
    """
    Creates a hyperopt tracing moodel function that sweep through search space
    Args:
        params: hyperparameters selected from search space
    Returns:
        hyperopt status and the loss metric value
    """
    mlflow.xgboost.autolog()

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
        model = xgb.XGBClassifier(**params)
        model = model.fit(X_train, y_train)

        # Predicting values for training and validation data
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Evaluating model metrics for training set predictions and validation set
        # Create train and val metrics dictionaries to make logging
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        train_acc = accuracy_score(y_train, y_train_pred).round(3)
        train_precision = precision_score(y_train, y_train_pred).round(3)
        train_recall = recall_score(y_train, y_train_pred).round(3)
        train_f1 = f1_score(y_train, y_train_pred).round(3)
        train_aucroc = roc_auc_score(y_train, y_train_pred_proba).round(3)
        training_metrics = {
            "accuracy": train_acc,
            "precision": train_precision,
            "recall": train_recall,
            "F1": train_f1, 
            "auc_roc": train_aucroc
        }
    
        train_metric_values = list(training_metrics.values())

        val_acc = accuracy_score(y_val, y_val_pred).round(3)
        val_precision = precision_score(y_val, y_val_pred).round(3)
        val_recall = recall_score(y_val, y_val_pred).round(3)
        val_f1 = f1_score(y_val, y_val_pred).round(3)
        val_aucroc = roc_auc_score(y_val, y_val_pred_proba).round(3)
        val_metrics = {
            "accuracy": val_acc,
            "precision": val_precision,
            "recall": val_recall,
            "F1": val_f1, 
            "auc_roc": val_aucroc
        }
    
        val_metric_values = list(val_metrics.values())
        
        # Logging model signature, class and name
        signature = infer_signature(X_train, y_val_pred)
        mlflow.xgboost.log_model(model, 'model', signature=signature)
        mlflow.set_tag('estimator_name', model.__class__.__name__)
        mlflow.set_tag('estimator_class', model.__class__)

        # Logging each metrics
        for name, metric in list(zip(metric_names, train_metric_values)):
            mlflow.log_metric(f'training_{name}', metric)
        for name, metric in list(zip(metric_names, val_metric_values)):
            mlflow.log_metric(f'val_{name}', metric)
        
        # Set the loss to -1*validation auc roc so fmin maximizes the it
        return {'status': STATUS_OK, 'loss': -1*val_metrics['auc_roc']}


trials = Trials() 

print("experiment id:")
print(EXPERIMENT_ID)

with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name='xgboost_models'):
    xgboost_best_params = fmin(
        fn=train_model,
        space=search_space, 
        algo=tpe.suggest, 
        trials=trials,
        max_evals=50
    )


# Querying mlflow api instead of using web UI. Sorting by validation aucroc and then getting top run for best run.
runs_df = mlflow.search_runs(experiment_ids=EXPERIMENT_ID, order_by=['metrics.validation_aucroc DESC'])
best_run = runs_df.iloc[0]
best_run_id = best_run['run_id']
best_artifact_uri = best_run['artifact_uri']
# Loading model from best run
best_model = mlflow.xgboost.load_model('runs:/' + best_run_id + '/model')

# Predicting and evaluating best model on holdout set
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred).round(3)
test_precision = precision_score(y_test, y_test_pred).round(3)
test_recall = recall_score(y_test, y_test_pred).round(3)
test_f1 = f1_score(y_test, y_test_pred).round(3)
test_aucroc = roc_auc_score(y_test, y_test_pred_proba).round(3)

print(f'Testing Accuracy: {test_accuracy}')
print(f'Testing Precision: {test_precision}')
print(f'Testing Recall: {test_recall}')
print(f'Testing F1: {test_f1}')
print(f'Testing AUCROC: {test_aucroc}')

model_details = mlflow.register_model(f'runs:/{best_run_id}/artifacts/model', 'BreastCancerClassification-XGBHP')

client = MlflowClient()
client.update_registered_model(
    name=model_details.name,
    description="""This model classifies breast cancer as malignant or benign given certain numerical features of cell nuclei such as 
    a) radius (mean of distances from center to points on the perimeter)
    b) texture (standard deviation of gray-scale values)
    c) perimeter
    d) area
    e) smoothness (local variation in radius lengths)
    f) compactness (perimeter^2 / area - 1.0)
    g) concavity (severity of concave portions of the contour)
    h) concave points (number of concave portions of the contour)
    i) symmetry
    j) fractal dimension ("coastline approximation" - 1)."""
)

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description='This model version is the first XGBoost model trained with HyperOpt for bayesian hyperparameter tuning.'
)

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage='Production'
)