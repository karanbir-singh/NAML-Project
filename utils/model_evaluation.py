import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_metrics(y_pred, y_true):
    """
    Utility function to compute some metrics about the model.

    Args: 
        y_pred (np.ndarray): the array of predicted labels.
        y_true (np.ndarray): the array of true labels.
    
    Returns:
        metrics: a dictionary containg the metrics
    """

    metrics = {}

    #True positive
    tp = np.sum((y_pred == 1) & (y_true == 1))

    #True negative
    tn = np.sum((y_pred == 0) & (y_true == 0))

    #False positive
    fp = np.sum((y_pred == 1) & (y_true == 0))

    #False negative 
    fn = np.sum((y_pred == 0) & (y_true == 1))

    #ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)

    metrics["precision"] = tp / (tp + fp)
    metrics["recall"] = tp / (tp + fn)
    metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["roc"] = {
        "fpr" : fpr,
        "tpr" : tpr
    }
    metrics["auc"] = auc_score
    metrics["mse"] = np.mean(y_true == y_pred)

    #Some other metrics related specifically to frauds that could be interesting to evaluate
    metrics["false_positive_rate"] = fp / (fp + tn),
    metrics["detection_rate"] = tp / (tp + fn),
    metrics["false_alarm_rate"] = fp / (fp + tp),
    
    metrics["total_transactions"] = len(y_true),
    metrics["fraud_transactions"] = sum(y_true),
    metrics["detected_frauds"] = tp,
    metrics["missed_frauds"] = fn,
    metrics["false_alarms"] = fp

    return metrics

feature_vectors = {
    "v1" : ["V1", "V5", "V7", "V8", "V11", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "Amount"],
    "v2" : ["V1", "V6", "V13", "V16", "V17", "V16", "V17", "V22", "V23", "V28", "Amount"],
    "v3" : ["V2", "V11", "V12", "V13", "V15", "V16", "V17", "V18", "V20", "V21", "V24", "V26", "Amount"],
    "v4" : ["V2", "V7", "V10", "V13", "V15", "V17", "V19", "V28", "V17", "Amount"],
    "v5" : ["Time", "V1", "V7", "V8", "V9", "V11", "V12", "V14", "V15", "V22", "V27", "V28", "Amount"],
    "all" : ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
}