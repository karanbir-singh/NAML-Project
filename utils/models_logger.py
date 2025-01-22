import mlflow
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from datetime import datetime
from tqdm import tqdm

def setup_tracking(experiment_name, push_to_server = False):
    """
        Sets up MLflow tracking with a specific experiment name.
        Creates a new experiment if it doesn't exist.

        Args:
            experiment_name: Name of the experiment.
            push_to_server: If True, logs will be sent to the remote server. Otherwise, logs will be saved locally.
    """

    if push_to_server:
        #Configure the remote tracking server URI
        mlflow.set_tracking_uri('mysql://sql7759036:a7VG3KkAV8@sql7.freesqldatabase.com:3306/sql7759036')
    else:
        #Configure local tracking
        mlflow.set_tracking_uri('file:.././mlruns')
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)

    return experiment_id

def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """
    A centralized method that allows you to compute metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for the positive class
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        #Standard classification metrics
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        
        #Fraud-specific metrics
        "false_positive_rate": fp / (fp + tn),
        "detection_rate": tp / (tp + fn),
        "false_alarm_rate": fp / (fp + tp),
        
        #Volume metrics
        "total_transactions": len(y_true),
        "fraud_transactions": sum(y_true),
        "detected_frauds": tp,
        "missed_frauds": fn,
        "false_alarms": fp
    }
    
    #ROC-AUC and PR-AUC can only be computed if probabilities are provided
    if y_pred_proba is not None:
        metrics.update({
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "pr_auc": average_precision_score(y_true, y_pred_proba)
        })
    
    return metrics

def log_fraud_model(experiment_name, model_name, y_true, y_pred, y_pred_proba = None, additional_params = None, 
                    note = None, push_to_server = False):
    """
    Logs fraud detection model metrics to MLflow.

    Args:
        experiment_name: Name of the experiment
        model_name: Name of the model
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        additional_params: Additional parameters to log (optional)
        model_artifact: Trained model to save (optional)
        note: A string to store notes about the model (optional)
        push_to_server: If True, logs will be sent to the remote server. Otherwise, logs will be saved locally.
    """

    #Log locally first
    setup_tracking(experiment_name, push_to_server = False)

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", nested=True) as run:
        mlflow.log_param("model_name", model_name)

        mlflow.log_param("on_server", push_to_server)

        if additional_params:
            mlflow.log_params(additional_params)
        
        #Log the note if provided
        if note:
            mlflow.log_param("note", note)
        
        #Compute and log metrics
        metrics = compute_metrics(y_true, y_pred, y_pred_proba)
        mlflow.log_metrics(metrics)
    
    #If push_to_server is True, log to the remote server
    if push_to_server:
        setup_tracking(experiment_name, push_to_server = True)

        with mlflow.start_run(run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", nested = True) as run:
            mlflow.log_param("model_name", model_name)

            mlflow.log_param("on_server", True)

            if additional_params:
                mlflow.log_params(additional_params)
            
            #Log the note if provided
            if note:
                mlflow.log_param("note", note)
            
            #Compute and log metrics
            mlflow.log_metrics(metrics)
            
            #Do not log the confusion matrix image to the server

def get_experiment_results(experiment_name, look_to_server=False):
    """
    Retrieves experiment results from MLflow.

    Args:
        experiment_name: Name of the experiment.
        look__to_server: If True, retrieves results from the remote server. Otherwise, retrieves from local logs.
    """

    setup_tracking(experiment_name, look_to_server)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) > 0:
        metrics_cols = [col for col in runs.columns if col.startswith('metrics.')]
        params_cols = [col for col in runs.columns if col.startswith('params.')]
        selected_cols = ['start_time', 'run_id'] + params_cols + metrics_cols
        
        results = runs[selected_cols].sort_values('start_time', ascending=False)
        
        #Format the 'start_time' column to show only up to seconds
        results['start_time'] = pd.to_datetime(results['start_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        results.columns = [col.replace('metrics.', '').replace('params.', '') for col in results.columns]
        return results
    
    return pd.DataFrame()

def sync_remote_to_local(experiment_name):
    """
        Syncs all runs from the remote server to the local tracking folder.

        Args:
            experiment_name: Name of the experiment to sync.
    """
    #Configure remote tracking
    setup_tracking(experiment_name, push_to_server = True)
    
    #Retrieve all runs from the remote server
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' does not exist on the remote server.")
        return
    
    print(f"Experiment ID: {experiment.experiment_id}")

    try:
        remote_runs = mlflow.search_runs(experiment_ids = [experiment.experiment_id])

        if remote_runs.empty:
            print(f"No runs found in experiment '{experiment_name}' on the remote server.")
            return
        
    except Exception as e:
        print(f"Error retrieving runs: {e}")
        return
    
    print(f"Found {len(remote_runs)} runs on the remote server. Syncing to local...")
    
    # Configure local tracking
    setup_tracking(experiment_name, push_to_server=False)
    
    # Recreate runs locally
    for _, run in tqdm(remote_runs.iterrows(), total = len(remote_runs), desc = "Syncing runs"):
        with mlflow.start_run(run_id = None, run_name = run['tags.mlflow.runName']):

            # Log parameters
            params = {key.replace("params.", ""): value for key, value in run.items() if key.startswith("params.")}
            mlflow.log_params(params)
            
            # Log metrics
            metrics = {key.replace("metrics.", ""): value for key, value in run.items() if key.startswith("metrics.")}
            mlflow.log_metrics(metrics)
            
            # Log tags
            tags = {key.replace("tags.", ""): value for key, value in run.items() if key.startswith("tags.")}
            for tag_key, tag_value in tags.items():
                mlflow.set_tag(tag_key, tag_value)
            
            # Set run ID
            mlflow.set_tag("original_run_id", run["run_id"])  # Reference original remote run ID
            
    print("Sync completed successfully!")