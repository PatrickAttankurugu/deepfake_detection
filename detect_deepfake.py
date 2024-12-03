import torch
import torch.nn as nn
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import re
from timm import create_model

class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder="tf_efficientnet_b7_ns"):
        super().__init__()
        self.encoder = create_model(encoder, pretrained=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        num_features = self.encoder.num_features
        self.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def prepare_image(image_path, size=380):
    """Prepare single image for inference"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(image, model="cnn")
    if not face_locations:
        raise ValueError("No faces detected in image")
    
    # Get the largest face
    top, right, bottom, left = max(face_locations, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))
    
    # Crop face with larger padding (50% instead of 30%)
    height, width = image.shape[:2]
    padding = int((bottom - top) * 0.5)
    top = max(0, top - padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    right = min(width, right + padding)
    face_image = image[top:bottom, left:right]
    
    # Apply transforms similar to training
    transform = A.Compose([
        A.Resize(size, size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)

def predict_deepfake(image_path, models, device):
    """Predict if an image is a deepfake using confident strategy"""
    try:
        image = prepare_image(image_path)
        image = image.to(device)
        
        predictions = []
        with torch.no_grad():
            for model in models:
                pred = torch.sigmoid(model(image)).cpu().numpy()[0][0]
                predictions.append(pred)
        
        # Use confident strategy from the codebase
        predictions = np.array(predictions)
        t = 0.8
        sz = len(predictions)
        fakes = np.count_nonzero(predictions > t)
        
        if fakes > sz // 2.5 and fakes > 11:
            confidence = np.mean(predictions[predictions > t])
        elif np.count_nonzero(predictions < 0.2) > 0.9 * sz:
            confidence = np.mean(predictions[predictions < 0.2])
        else:
            confidence = np.mean(predictions)
            
        individual_preds = {f"Model_{i+1}": f"{pred:.4f}" 
                          for i, pred in enumerate(predictions)}
        
        return {
            'is_fake': confidence > 0.5,
            'confidence': confidence,
            'individual_predictions': individual_preds
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def load_models(weights_folder):
    """Load models on CPU"""
    models = []
    device = torch.device('cpu')
    
    # List all weight files in the weights directory
    model_files = [f for f in os.listdir(weights_folder) 
                  if f.startswith('final_') and not f.endswith('.txt')]
    
    print(f"Loading {len(model_files)} models...")
    for model_file in model_files:
        print(f"Loading {model_file}...")
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
        
        # Load checkpoint
        checkpoint = torch.load(f"{weights_folder}/{model_file}", 
                              map_location=device)
        
        # Get state dict from checkpoint
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Remove 'module.' prefix if it exists
        state_dict = {re.sub("^module.", "", k): v for k, v in state_dict.items()}
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        models.append(model)
    
    return models, device

if __name__ == "__main__":
    # Load models
    print("Initializing models...")
    weights_folder = "weights"
    models, device = load_models(weights_folder)
    
    # Get image path from user
    image_path = input("\nEnter the path to your image: ")
    
    # Run inference
    print("\nProcessing image...")
    result = predict_deepfake(image_path, models, device)
    
    if result:
        print(f"\nDeepfake Detection Results:")
        print(f"{'=' * 50}")
        print(f"Verdict: {'FAKE' if result['is_fake'] else 'REAL'}")
        print(f"Confidence Score: {result['confidence']:.4f}")
        print(f"\nIndividual Model Predictions:")
        for model, pred in result['individual_predictions'].items():
            print(f"{model}: {pred}")