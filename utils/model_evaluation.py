import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd

def compute_metrics(y_true, y_pred, metrics_df = None, dataset_label = ''):
    """
        Computes and print metrics tp, tn, fp, fn, AC, rc, PC, F1

        Parameters:
        predictions: ndarray - predictions of samples obtained with a model
        y_true: ndarray - true labels of the samples
        metrics_df: DataFrame - DataFrame to which the computed statistics have to be put
        dataset_label: str - label identifying the belonging of the statistics to its dataset

        Returns:
        DataFrame - DataFrame containing the statistics contained in the parameter metrics_df
                    plus the statistics computed on the new predictions
    """
    
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)

    ac = accuracy(y_true, y_pred)
    rc = recall(y_true, y_pred)
    pr = precision(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    total_transaction = len(y_true)
    fraud_transaction = sum(y_true)

    columns = ['set of features', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'recall', 'precision', 'f1-score', 'total transaction', 'fraud transactions']

    if metrics_df is None:
        metrics_df = pd.DataFrame([[dataset_label, tp, tn, fp, fn, ac, rc, pr, f1, total_transaction, fraud_transaction]], columns = columns)
    else:
        metrics_df = pd.concat([metrics_df, pd.DataFrame([[dataset_label, tp, tn, fp, fn, ac, rc, pr, f1, total_transaction, fraud_transaction]], columns = columns)], ignore_index = True)

    return metrics_df

def compute_roc_auc(y_true, y_prob):
    """
        Computes ROC-AUC related metrics
    """

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score

def confusion_matrix(true_labels, pred_labels):
    """
        Computes the confusion matrix

        Parameters:
        true_labels: ndarray - correct values of the samples' class'
        pred_labels: ndarray - predicted values of the samples' class'

        Returns:
        TP: float - true positives - attacks classified accurately as attacks
        TN: float - true negatives - normal transactions accurately classified as normal
        FP: float - false positives - normal traffic incorrectly classified as attacks
        FN: float - false negatives - attacks incorrectly classified as normal
    """

    tp = np.sum(np.logical_and(pred_labels == 1., true_labels == 1.))
    tn = np.sum(np.logical_and(pred_labels == 0., true_labels == 0.))
    fp = np.sum(np.logical_and(pred_labels == 1., true_labels == 0.))
    fn = np.sum(np.logical_and(pred_labels == 0., true_labels == 1.))

    return tp, tn, fp, fn

def accuracy(true_labels, pred_labels):
    """
        Computes the accuracy of the predictions

        Parameters:
        true_labels: ndarray - correct values of the samples' class'
        pred_labels: ndarray - predicted values of the samples' class'

        Returns:
        float - accuracy of the artificial neural network, namely the number of samples
                correctly classified divided by the total number of samples
    """
    tp, tn, _, _ = confusion_matrix(true_labels, pred_labels)

    return (tn + tp) / len(pred_labels)

def recall(true_labels, pred_labels):
    """
        Computes the recall (or sensitivity) of the predictions

        Parameters:
        true_labels: ndarray - correct values of the samples' class'
        pred_labels: ndarray - predicted values of the samples' class'

        Returns:
        float - recall of the artificial neural network,
                namely the percentage of positive predictions (true positive rate),
                out of the total positive
    """
    tp, _, _, fn = confusion_matrix(true_labels, pred_labels)
    return tp / (tp + fn)

def precision(true_labels, pred_labels):
    """
        Computes the precision of the predictions

        Parameters:
        true_labels: ndarray - correct values of the samples' class'
        pred_labels: ndarray - predicted values of the samples' class'

        Returns:
        float - precision of the artificial neural network, namely the percentage of truly positive,
                out of all positive predicted
    """
    tp, _, fp, _ = confusion_matrix(true_labels, pred_labels)

    return tp / (tp + fp)

def f1_score(true_labels, pred_labels):
    """
        Computes the F1 Score of the predictions

        Parameters:
        true_labels: ndarray - correct values of the samples' class'
        pred_labels: ndarray - predicted values of the samples' class'

        Returns:
        float - f1 score of the artificial neural network, namely the harmonic mean of precision and recall.
                It takes both false positive and false negatives into account
    """
    rc = recall(true_labels, pred_labels)
    pr = precision(true_labels, pred_labels)

    return 2 * pr * rc / (pr + rc)

feature_vectors = {
    "v1" : ["V1", "V5", "V7", "V8", "V11", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "Amount"],
    "v2" : ["V1", "V6", "V13", "V16", "V17", "V22", "V23", "V28", "Amount"],
    "v3" : ["V2", "V11", "V12", "V13", "V15", "V16", "V17", "V18", "V20", "V21", "V24", "V26", "Amount"],
    "v4" : ["V2", "V7", "V10", "V13", "V15", "V17", "V19", "V28", "Amount"],
    "v5" : ["Time", "V1", "V7", "V8", "V9", "V11", "V12", "V14", "V15", "V22", "V27", "V28", "Amount"],
    "v6" : ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"],
    "v7" : ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
}