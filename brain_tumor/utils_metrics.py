import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

def brier_loss(y_true, prob):
    return ((y_true - prob)**2).mean().cpu()

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()
# # Example usage
# y_true = [1, 0, 1, 2, 2, 0, 1, 1, 2, 0]
# y_pred = [1, 0, 1, 2, 0, 0, 1, 2, 2, 0]
# labels = [0, 1, 2]
# plot_confusion_matrix(y_true, y_pred, labels=labels)

def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.show()

def evaluate_model(val_labels, val_probs):
    print("# Base Model Evaluation")
    print("Accuracy:", accuracy_score(val_labels, val_probs>0.5))
    print("F1 Score:", f1_score(val_labels, val_probs>0.5))
    print("Average Precision Score:", average_precision_score(val_labels, val_probs))
    plot_confusion_matrix(
        ["Yes" if i==1 else "No" for i in val_labels], 
        ["Yes" if i==1 else "No" for i in (val_probs>0.5)],
        labels=["No", "Yes"]
    )
    plot_precision_recall_curve(val_labels, val_probs)
    
    print("\n# Calibration Evaluation")
    print("Log Loss:", log_loss(val_labels, val_probs))
    print("Brier Loss:", brier_score_loss(val_labels, val_probs))

    prob_true_uncal, prob_pred_uncal = calibration_curve(val_labels, val_probs, n_bins=10)
    plt.plot(prob_pred_uncal, prob_true_uncal, "s-")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.show()
