import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, cbar=False, ax=ax)
    ax.set(title=title, ylabel='True label', xlabel='Predicted label')

    # Save plot
    plt.savefig('confusion_matrix.png')
    plt.close()

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_outputs = []  # To store raw outputs for AUC-ROC
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())  # Store raw outputs

    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc_roc = roc_auc_score(all_labels, all_outputs)  # Use aggregated raw outputs for AUC-ROC
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc
    }

    print(f'Test Accuracy: {accuracy * 100}%')
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, classes=[0, 1])

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    return accuracy, metrics
