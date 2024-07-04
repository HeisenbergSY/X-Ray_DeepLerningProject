import yaml
import torch
import time
import numpy as np
import random
import platform
from data_loader import get_data_loaders, get_k_fold_data_loaders
from model import get_model  # Import the dynamic model selection function
from train import train_model, validate_model  # Correctly import validate_model
from test import test_model
from plot_utils import plot_accuracies, plot_confusion_matrix
from logger import log_final_results
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    config = load_config()

    model_name = config["model_name"]  # Read model name from config

    # Set random seed for reproducibility
    if config["use_random_seed"]:
        seed = random.randint(0, 10000)
        print(f"Using random seed: {seed}")
    else:
        seed = config["seed"]
        print(f"Using fixed seed: {seed}")
    set_random_seed(seed)

    train_loader, original_train_loader, val_loader, test_loader, class_names = get_data_loaders(
        config["train_dir"], config["val_dir"], config["test_dir"], batch_size=config["batch_size"], model_name=model_name
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    best_val_loss = float('inf')
    best_hyperparams = None
    best_model_state_dict = None
    execution_start_time = time.time()

    results = []

    if config["use_k_fold"]:
        k_fold_loaders, class_names = get_k_fold_data_loaders(
            config["train_dir"], batch_size=config["batch_size"], k=config["k_folds"]
        )

        for lr in config["learning_rates"]:
            for patience in config["patience_values"]:
                fold_val_losses = []
                fold_val_accuracies = []

                for fold, (train_loader, val_loader) in enumerate(k_fold_loaders):
                    print(f'Fold {fold+1}/{config["k_folds"]}, Learning Rate: {lr}, Patience: {patience}')

                    model = get_model(model_name)  # Use dynamic model selection
                    model = model.to(device)

                    start_time = time.time()
                    training_accuracies, validation_accuracies, training_losses, validation_losses = train_model(
                        model, train_loader, val_loader, device, num_epochs=config["num_epochs"], learning_rate=lr, patience=patience
                    )
                    fold_val_accuracy, fold_val_loss = validate_model(model, val_loader, device)
                    fold_val_losses.append(fold_val_loss)
                    fold_val_accuracies.append(fold_val_accuracy)
                    execution_time = time.time() - start_time

                    if fold_val_loss < best_val_loss:
                        best_val_loss = fold_val_loss
                        best_val_accuracy = fold_val_accuracy
                        best_hyperparams = {"learning_rate": lr, "patience": patience}
                        best_model_state_dict = model.state_dict()

                avg_val_loss = np.mean(fold_val_losses)
                avg_val_accuracy = np.mean(fold_val_accuracies)

                results.append({
                    "learning_rate": lr,
                    "patience": patience,
                    "avg_val_loss": avg_val_loss,
                    "avg_val_accuracy": avg_val_accuracy
                })

                print(f'Learning Rate: {lr}, Patience: {patience}, Average Validation Loss: {avg_val_loss}, Average Validation Accuracy: {avg_val_accuracy * 100}%')
    else:
        for lr in config["learning_rates"]:
            for patience in config["patience_values"]:
                print(f'Learning Rate: {lr}, Patience: {patience}')

                model = get_model(model_name)  # Use dynamic model selection
                model = model.to(device)

                start_time = time.time()
                training_accuracies, validation_accuracies, training_losses, validation_losses = train_model(
                    model, train_loader, val_loader, device, num_epochs=config["num_epochs"], learning_rate=lr, patience=patience
                )
                val_accuracy, val_loss = validate_model(model, val_loader, device)
                execution_time = time.time() - start_time

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_accuracy = val_accuracy
                    best_hyperparams = {"learning_rate": lr, "patience": patience}
                    best_model_state_dict = model.state_dict()

                results.append({
                    "learning_rate": lr,
                    "patience": patience,
                    "avg_val_loss": val_loss,
                    "avg_val_accuracy": val_accuracy
                })

                print(f'Learning Rate: {lr}, Patience: {patience}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy * 100}%')

    # Save the best model after all folds and hyperparameters have been evaluated
    if best_model_state_dict is not None:
        best_model_path = f'best_model_{model_name}.pth'
        torch.save(best_model_state_dict, best_model_path)
        print(f'Saving the best model with validation loss: {best_val_loss}')

    # Evaluate on test set using the best model
    model = get_model(model_name)
    model.load_state_dict(best_model_state_dict)
    model = model.to(device)
    test_accuracy, test_metrics = test_model(model, test_loader, device)

    # Plot accuracies
    plot_accuracies(training_accuracies, validation_accuracies, model_name)

    # Compute confusion matrix
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            predicted = (outputs >= 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    plot_confusion_matrix(all_labels, all_predictions, classes=class_names, model_name=model_name, normalize=True)

    # Log the results
    with open("hyperparameter_tuning_results.txt", "w") as f:
        for result in results:
            f.write(f"Learning Rate: {result['learning_rate']}, Patience: {result['patience']}, "
                    f"Average Validation Loss: {result['avg_val_loss']}, "
                    f"Average Validation Accuracy: {result['avg_val_accuracy'] * 100}%\n")

        f.write(f"\nBest Hyperparameters: {best_hyperparams}, Best Validation Loss: {best_val_loss}, Best Validation Accuracy: {best_val_accuracy * 100}%\n")

    print(f'Best Hyperparameters: {best_hyperparams}, Best Validation Loss: {best_val_loss}, Best Validation Accuracy: {best_val_accuracy * 100}%')

    # Final log
    execution_time = time.time() - execution_start_time
    log_final_results(config, model_name, execution_time, config["num_epochs"], best_val_accuracy, test_metrics, device)

if __name__ == '__main__':
    main()
