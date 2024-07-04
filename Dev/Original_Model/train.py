import torch
from torch import nn, optim

def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_accuracies = []
    validation_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = correct_train / total_train
        val_accuracy = validate_model(model, val_loader, device)
        
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy * 100}%, Validation Accuracy: {val_accuracy * 100}%')

    return training_accuracies, validation_accuracies

def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
