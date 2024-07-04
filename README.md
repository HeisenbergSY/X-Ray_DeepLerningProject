# Chest X-Ray Classification

This project classifies chest X-ray images into two categories: Normal and Pneumonia. The project involves data augmentation, handling class imbalance, 
model training, evaluation, and inference.

## Dataset

The dataset used for this project consists of chest X-ray images categorized into two classes: Normal and Pneumonia. The dataset can be downloaded from:
[Chest X-Ray Dataset](https://drive.google.com/drive/folders/1N9D68Uj6Y3R8_iYAE_dnP9J5BXUiDXRy).

## Installation

Requirements
To install the required libraries, use the requirements.txt file provided in the repository. Run the following command:
pip install -r requirements.txt

## Usage

Download and Extract Dataset:
Download the dataset from the link provided above and extract it to your desired location. Make sure the directory structure looks like this:

dataset/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
        
## Config File:

The config.yaml file contains the configuration for the training process. Below is an example configuration:

- model_name: ResNet50Binary  # Specify the model name (e.g., MobileNetV3Binary, VGG16Binary)
- train_dir: path/to/train
- val_dir: path/to/val
- test_dir: path/to/test
- batch_size: 32
- num_epochs: 10
- learning_rates:
-   - 0.001
-   - 0.0001
- patience_values:
-   - 3
-   - 4
- use_k_fold: false  # Enable or disable k-fold cross-validation
- k_folds: 3
- use_random_seed: true  # Specify whether to use a random seed
- seed: 42  # This will be ignored if use_random_seed is true

model_name: Specify the model architecture to be used (e.g., MobileNetV3Binary, VGG16Binary, ResNet50Binary).
train_dir, val_dir, test_dir: Paths to the training, validation, and test directories.
batch_size: Batch size for training and validation.
num_epochs: Number of epochs for training.
learning_rates: List of learning rates to try during hyperparameter tuning.
patience_values: List of patience values for early stopping during hyperparameter tuning.
use_k_fold: Enable or disable k-fold cross-validation.
k_folds: Number of folds for k-fold cross-validation.
use_random_seed: Specify whether to use a random seed.
seed: Random seed value (ignored if use_random_seed is true).

## Training:

To start the training process, run the run.py script:
python run.py
This will train the model using the specified configuration and save the best model based on validation loss.

## Evaluation:

After training, the script evaluates the best model on the test set and logs the results. The evaluation includes accuracy, precision, recall, F1-score, and AUC-ROC. It also plots the accuracy over epochs and the confusion matrix.

## Code Structure

- data_loader.py: Handles data loading, transformations, and class distribution plotting.
- model.py: Contains the model architectures.
- train.py: Handles the training process and early stopping.
- test.py: Handles model evaluation on the test set.
- plot_utils.py: Contains functions for plotting accuracies and confusion matrices.
- logger.py: Logs the final results to a text file.
- config.yaml: Configuration file for the training process.
- run.py: Main script to run the training and evaluation pipeline.

## Customization

### Adding New Models
To add a new model, define the model architecture in model.py and specify the model name in the config.yaml file.

### Data Augmentation
Data augmentation techniques such as random rotations, flips, and histogram equalization are applied in data_loader.py. You can customize these transformations in the get_transforms function.

### Early Stopping
Early stopping is implemented in train.py with configurable patience values specified in the config.yaml file.

### K-Fold Cross-Validation
K-fold cross-validation is implemented in data_loader.py. You can enable it by setting use_k_fold to true in config.yaml and specifying the number of folds with k_folds.

## Results

The script logs the best hyperparameters and evaluation metrics. It also saves the best model with the filename format best_model_<model_name>.pth.
Furthermore a confussion matrix and an accuracy plot will be created after each training.

## Contributing

Feel free to open issues or submit pull requests if you want to contribute to this project.
