import torch
from torchvision import transforms
from PIL import Image
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox
from model import get_model  # Ensure this is the function to get the correct model architecture
from inference import infer  # Import the infer function

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

def choose_images():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames()
    root.destroy()
    return file_paths

def main():
    config = load_config()
    model_name = config["model_name"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the best model
    model_path = filedialog.askopenfilename(title="Select Model File", filetypes=(("PyTorch Model", "*.pth"),))
    if not model_path:
        messagebox.showerror("Error", "Model file not selected")
        return

    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Choose images for inference
    image_paths = choose_images()
    if not image_paths:
        messagebox.showerror("Error", "Image files not selected")
        return

    for image_path in image_paths:
        image = load_image(image_path).to(device)
        
        # Perform inference
        prediction, probability = infer(model, image)

        # Display the prediction with probability
        predicted_class = "Normal" if prediction == 0 else "Pneumonia"
        messagebox.showinfo("Prediction", f"The predicted class for {image_path} is: {predicted_class} with probability {probability:.2f}")

if __name__ == "__main__":
    main()
