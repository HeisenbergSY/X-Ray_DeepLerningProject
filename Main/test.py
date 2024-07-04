import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    accuracy = correct / total
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    precision = precision_score(all_labels, all_predictions.round())
    recall = recall_score(all_labels, all_predictions.round())
    f1 = f1_score(all_labels, all_predictions.round())
    auc_roc = roc_auc_score(all_labels, all_predictions)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc
    }

    print(f'Test Accuracy: {accuracy * 100}%')
    return accuracy, metrics
