import torch
import platform

def log_final_results(config, model_name, execution_time, num_epochs, final_val_accuracy, test_metrics, device_info):
    log_filename = f"final_results_{model_name}.txt"
    augmentations = [
        "RandomRotation(30)",
        "RandomHorizontalFlip()",
        "RandomResizedCrop(224, scale=(0.8, 1.0))",
        "HistogramEqualization()",
        "RandomAffine(degrees=0, translate=(0.1, 0.1))",
        "RandomVerticalFlip()"
    ]
    
    with open(log_filename, 'w') as f:
        f.write(f"Learning Rate: {config['learning_rates'][0]}\n")  # Assuming the best learning rate is the first one in the list
        f.write(f"Data Augmentations: {augmentations}\n")
        f.write(f"Model Used: {model_name}\n")
        f.write(f"Execution Time (seconds): {execution_time}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Final Validation Accuracy: {final_val_accuracy * 100}\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy']}\n")
        f.write(f"Test Precision: {test_metrics['precision']}\n")
        f.write(f"Test Recall: {test_metrics['recall']}\n")
        f.write(f"Test F1-score: {test_metrics['f1_score']}\n")
        f.write(f"Test AUC-ROC: {test_metrics['auc_roc']}\n")
        f.write(f"Hardware Profile:\n")
        f.write(f"  Platform: {platform.platform()}\n")
        f.write(f"  Processor: {platform.processor()}\n")
        f.write(f"  RAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)} GB\n")
        f.write(f"  CUDA Available: {torch.cuda.is_available()}\n")
        f.write(f"  CUDA Device Count: {torch.cuda.device_count()}\n")
        f.write(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}\n")
