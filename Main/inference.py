import torch

def infer(model, image, threshold=0.5):
    model.eval()
    with torch.no_grad():
        output = model(image)
        print("Model output before rounding:", output)  # Debugging: print raw output
        probability = output.item()  # Extract the probability
        prediction = (output >= threshold).float()  # Thresholding at specified value for binary classification
    return prediction.item(), probability
