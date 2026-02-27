from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

def plot_roc_pr(model, X, y):

    y_proba = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_proba)
    precision, recall, _ = precision_recall_curve(y, y_proba)

    return fpr, tpr, precision, recall

def evaluate_model(model, x_test, y_test):

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC:", average_precision_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))