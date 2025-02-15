import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import  classification_report, ConfusionMatrixDisplay

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

def evaluate_model(val_labels, val_probs, class_names):
    """
    Evaluates a multiclass model using accuracy, F1-score, precision-recall, and calibration metrics.

    Parameters:
        val_labels (torch.Tensor or np.array): True labels (shape: [num_samples]).
        val_probs (torch.Tensor or np.array): Predicted probabilities (shape: [num_samples, num_classes]).
        class_names (list): List of class names corresponding to the model's output classes.
    """
    
    # Convert tensors to numpy (if necessary)
    val_labels = val_labels.cpu().numpy() if isinstance(val_labels, torch.Tensor) else np.array(val_labels)
    val_probs = val_probs.cpu().numpy() if isinstance(val_probs, torch.Tensor) else np.array(val_probs)
    
    # Get predicted classes (argmax over probabilities)
    val_preds = np.argmax(val_probs, axis=1)

    # Compute Accuracy and F1-Score
    print("# Base Model Evaluation")
    print("Accuracy:", accuracy_score(val_labels, val_preds))
    print("F1 Score (macro):", f1_score(val_labels, val_preds, average="macro"))
    print("F1 Score (weighted):", f1_score(val_labels, val_preds, average="weighted"))

    # Classification Report (Precision, Recall, F1 per class)
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(val_labels, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    # Precision-Recall Curves per class
    print("\n# Precision-Recall Curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(val_labels == i, val_probs[:, i])
        ax.plot(recall, precision, label=f"Class {class_name}")
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.show()

    # Average Precision Score (Mean Average Precision)
    print("Average Precision Score (macro):", average_precision_score(val_labels, val_probs, average="macro"))

    # Calibration Evaluation
    print("\n# Calibration Evaluation")
    print("Log Loss:", log_loss(val_labels, val_probs))
    print("Brier Score Loss (macro):", np.mean([brier_score_loss(val_labels == i, val_probs[:, i]) for i in range(len(class_names))]))

    # Calibration Curve Plot
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        prob_true, prob_pred = calibration_curve(val_labels == i, val_probs[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"Class {class_name}")

    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()
