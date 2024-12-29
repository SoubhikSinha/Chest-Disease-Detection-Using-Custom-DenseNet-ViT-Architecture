# Let us first import all the necessary libraries required for this project

import tensorflow as tf
import torch
import cv2
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc


# Later on, as per requirement, more libraries wil be imported

import gradio as gr

def adjust_brightness_contrast(image, alpha=1.2, beta=50):
    """
    Adjusting brightness and contrast of the image.
    Parameters:
        - image: Input image (numpy array).
        - alpha: Contrast control [1.0-3.0].
        - beta: Brightness control [0-100].
    Returns:
        - Adjusted image.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_histogram_equalization(image):
    """Applying histogram equalization to enhance contrast."""
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(eq_channels)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applying CLAHE for local contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(image)
    clahe_channels = [clahe.apply(ch) for ch in channels]
    return cv2.merge(clahe_channels)

def apply_gaussian_blur(image, kernel_size=(3, 3)):
    """Applying Gaussian blur for denoising."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_sharpening(image):
    """Applying edge enhancement using a sharpening filter."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def normalize_image(image):
    """Normalizing the image to zero mean and unit variance."""
    image = (image - np.mean(image)) / np.std(image)
    return image

def resize_image(image, width, height):
    """Resizing the image to the desired dimensions with anti-aliasing."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def preprocess_single_image(pil_image, img_height=224, img_width=224):
    """
    Preprocessing a single image as per the training pipeline.
    Parameters:
        - pil_image: Input PIL image.
        - img_height, img_width: Dimensions to resize the image.
    Returns:
        - Preprocessed image tensor.
    """
    # Converting PIL image to numpy array
    image = np.array(pil_image)

    # Ensuring the image is in RGB format
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Applying preprocessing steps
    image = apply_histogram_equalization(image)
    image = apply_clahe(image)
    image = apply_gaussian_blur(image)
    image = apply_sharpening(image)
    image = adjust_brightness_contrast(image, alpha=1.2, beta=50)

    # Resizing and normalization
    image = resize_image(image, img_width, img_height)
    image = normalize_image(image)

    # Converting to PIL image and applying transformations
    image = Image.fromarray(image.astype(np.uint8))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  # Adding batch dimension
    return image_tensor


# Detecting GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.models import densenet121
from torchvision.transforms import Resize

# Simplified ViT-like transformer module
class SimpleViT(nn.Module):
    def __init__(self, input_dim, num_heads, mlp_dim, num_layers):
        super(SimpleViT, self).__init__()
        # Reduced TransformerEncoder layer complexity
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=0.1
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        # Flattening the spatial dimensions
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # Reshaping for transformer
        for block in self.transformer_blocks:
            x = block(x)
        x = x.permute(1, 2, 0).reshape(B, C, H, W)  # Restoring the original shape
        return x


# Adjusted Hybrid DenseNet + Simplified ViT Architecture
class LightweightHybridDenseNetViT(nn.Module):
    def __init__(self):
        super(LightweightHybridDenseNetViT, self).__init__()

        # Loading a lighter DenseNet backbone
        self.densenet = densenet121(pretrained=False)  # Base DenseNet backbone

        # Reducing the output channels from DenseNet to smaller dimensions
        self.conv_reduce = nn.Conv2d(1024, 64, kernel_size=1)  # Fewer channels

        # ViT processing module with reduced complexity
        self.vit = SimpleViT(input_dim=64, num_heads=2, mlp_dim=128, num_layers=1)

        # Task-specific classification heads
        self.fc_pneumonia = nn.Linear(64, 1)  # Binary classification (Pneumonia)
        self.fc_tuberculosis = nn.Linear(64, 1)  # Binary classification (Tuberculosis)
        self.fc_lung_cancer = nn.Linear(64, 4)  # Multi-class output (Lung Cancer)

    def forward(self, x):
        # Extracting DenseNet features
        x = self.densenet.features(x)  # Extracting DenseNet feature maps
        x = self.conv_reduce(x)  # Reducing the number of feature channels

        # Passing through simplified ViT module
        x = self.vit(x)

        # Applying Global Average Pooling (GAP)
        x = x.mean(dim=[2, 3])  # Pooling across spatial dimensions

        # Task-specific classification
        pneumonia_output = torch.sigmoid(self.fc_pneumonia(x))  # Binary sigmoid output
        tuberculosis_output = torch.sigmoid(self.fc_tuberculosis(x))  # Binary sigmoid output
        lung_cancer_output = self.fc_lung_cancer(x)  # Multi-class logits

        return pneumonia_output, tuberculosis_output, lung_cancer_output
    

# Loading the saved model
img_size = 224  # Matching the dimensions used during training
patch_size = 8


model = LightweightHybridDenseNetViT().to(device)

model.load_state_dict(torch.load("model_FINAL.pth", map_location=device))  # Mapping to the correct device
model.to(device)  # Moving the model to GPU/CPU
model.eval()  # Setting to evaluation mode


# Function to pre-process the image and perform inference
def predict_image(image):
    """
    Predicts the probabilities of Pneumonia, TB, and Lung Cancer from the input image.
    """
    # Preprocessing the image
    image_tensor = preprocess_single_image(image, img_height=224, img_width=224)
    image_tensor = image_tensor.to(device)
    
    # Performing inference
    with torch.no_grad():
        pneumonia_output, tb_output, lung_cancer_output = model(image_tensor)
    
    # Getting the probabilities
    pneumonia_prob = pneumonia_output.item()
    tb_prob = tb_output.item()
    lung_cancer_probs = F.softmax(lung_cancer_output, dim=1).squeeze().tolist()

    # Class names for lung cancer
    lung_cancer_classes = [
        "adenocarcinoma_left.lower.lobe",
        "large.cell.carcinoma_left.hilum",
        "NORMAL",
        "squamous.cell.carcinoma_left.hilum"
    ]
    
    # Preparing the result as a dictionary
    result = {
        "Pneumonia Probability": f"{pneumonia_prob:.4f}",
        "TB Probability": f"{tb_prob:.4f}",
        "Lung Cancer Probabilities": {class_name: f"{prob:.4f}" for class_name, prob in zip(lung_cancer_classes, lung_cancer_probs)}
    }
    
    return result

# Gradio Interface
iface = gr.Interface(fn=predict_image, 
                     inputs=gr.Image(type="pil"), 
                     outputs=gr.JSON(),
                     title="Probabilistic Lung Disease Detection",
                     description="An AI-powered tool that analyzes and predicts probabilities for lung diseases, including Pneumonia, Tuberculosis, and Lung Cancer.")

iface.launch()