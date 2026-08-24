import torch
from torchvision import models, transforms
from PIL import Image
import requests
import json
import os

# Define the image path
image_path = "hen.jpg"

# Check if the image exists
try:
    img = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: Could not find the image {image_path}")
    exit()

# Load the pre-trained ResNet50 model
print("Loading pre-trained ResNet50 model...")
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Preprocess the input image
print ("Preprocessing image...")
input_tensor = preprocess(img).unsqueeze_(0)

# Perform inference
print("Classifying image...")
with torch.no_grad():
    output = model(input_tensor)

# Download ImageNet class labels if not already available
labels_url= "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels_file = "imagenet_classes.json"

if not os.path.exists(labels_file):
    print("Downloading imagenet class labels...")
    with open(labels_file, "w") as f: f.write(requests.get(labels_url).text)

# Load the class labels
with open(labels_file, "r") as f:
    labels = json.load(f)

# Get the top predicted class
_, predicted_idx = output.max(1)
predicted_class = labels[predicted_idx.item()]

# Print the classifications result
print(f"The image is classified as: {predicted_class}")