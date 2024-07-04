import torch
import time
import platform
import psutil
import matplotlib.pyplot as plt
from data_loader import get_data_loaders
from model import MobileNetV3Binary
from train import train_model, validate_model
from test import test_model

def main():
    train_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\train'
    val_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\val'
    test_dir = r'C:\Users\thepr\DeepLerningProject\chest_xray\test'

    train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV3Binary()
    model = model.to(device)

    learning_rate = 0.001
    num_epochs = 10

    start_time = time.time()
    training_accuracies, validation_accuracies = train_model(model, train_loader, val_loader, device, num_epochs=num_epochs, learning_rate=learning_rate)
    final_val_accuracy = validate_model(model, val_loader, device)
    test_accuracy, test_metrics = test_model(model, test_loader, device)
    execution_time = time.time() - start_time

    log_results(learning_rate, model.__class__.__name__, execution_time, final_val_accuracy, test_accuracy, test_metrics, device)
    plot_accuracies(training_accuracies, validation_accuracies)

def plot_accuracies(training_accuracies, validation_accuracies):
    epochs = range(1, len(training_accuracies) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

def log_results(learning_rate, model_name, execution_time, final_val_accuracy, test_accuracy, test_metrics, device):
    log_data = {
        "Learning Rate": learning_rate,
        "Model Used": model_name,
        "Execution Time (seconds)": execution_time,
        "Final Validation Accuracy": final_val_accuracy,
        "Test Accuracy": test_accuracy,
        "Test Precision": test_metrics['precision'],
        "Test Recall": test_metrics['recall'],
        "Test F1-score": test_metrics['f1'],
        "Test AUC-ROC": test_metrics['auc_roc'],
        "Hardware Profile": {
            "Platform": platform.platform(),
            "Processor": platform.processor(),
            "RAM": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
            "CUDA Available": torch.cuda.is_available(),
            "CUDA Device Count": torch.cuda.device_count(),
            "CUDA Device Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }
    with open("training_log.txt", "w") as log_file:
        for key, value in log_data.items():
            if isinstance(value, dict):
                log_file.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    log_file.write(f"  {sub_key}: {sub_value}\n")
            else:
                log_file.write(f"{key}: {value}\n")

if __name__ == '__main__':
    main()
