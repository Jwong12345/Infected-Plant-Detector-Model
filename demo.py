"""
Demo script to test the complete pipeline:
1. ResNet50 classifies image as healthy/diseased
2. If diseased, U-Net segments the diseased regions
3. OpenCV highlights the segmented regions
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from unet_model import UNet

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

unet_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_models(resnet_path='best_resnet50_plant_model.pth', unet_path='best_unet_model.pth'):
    """Load both ResNet50 and U-Net models"""
    print("Loading ResNet50 model...")
    resnet_model = models.resnet50(weights=None)
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, 2)
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    
    print("Loading U-Net model...")
    unet_model = UNet(n_channels=3, n_classes=1)
    unet_model.load_state_dict(torch.load(unet_path, map_location=device))
    unet_model = unet_model.to(device)
    unet_model.eval()
    
    print("Models loaded successfully!")
    return resnet_model, unet_model

def classify_image(resnet_model, image_path):
    """Classify image using ResNet50"""
    image = Image.open(image_path).convert('RGB')
    input_tensor = resnet_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = resnet_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_names = ['Diseased', 'Healthy']
    is_diseased = predicted_class == 0
    return is_diseased, confidence, class_names[predicted_class], image.size

def segment_image(unet_model, image):
    """Segment diseased regions using U-Net"""
    original_size = image.size
    input_tensor = unet_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mask = unet_model(input_tensor)
        mask = mask.squeeze().cpu().numpy()
    
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
    return mask_resized

def highlight_regions(image, mask, threshold=0.5):
    """Use OpenCV to highlight diseased regions"""
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Create binary mask
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create overlay
    overlay = img_array.copy()
    highlighted = img_array.copy()
    
    # Draw filled contours
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)  # Red fill
    highlighted = cv2.addWeighted(overlay, 0.4, highlighted, 0.6, 0)
    
    # Draw outlines
    cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)
    
    return highlighted, len(contours)

def process_image(resnet_model, unet_model, image_path, output_path=None):
    """Complete pipeline"""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    # Step 1: Classification
    is_diseased, confidence, class_name, original_size = classify_image(resnet_model, image_path)
    print(f"✓ Classification: {class_name} (confidence: {confidence:.2%})")
    
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if not is_diseased:
        print("✓ Image is healthy - no segmentation needed")
        cv2.putText(img_array, f'Healthy ({confidence:.1%})', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, img_array)
            print(f"✓ Result saved to: {output_path}")
        
        return img_array
    
    # Step 2: Segmentation
    print("✓ Image is diseased - performing segmentation...")
    mask = segment_image(unet_model, image)
    
    # Step 3: Highlighting
    highlighted, num_regions = highlight_regions(image, mask)
    
    # Add labels
    cv2.putText(highlighted, f'{class_name} ({confidence:.1%})', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(highlighted, f'Diseased Regions: {num_regions}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    print(f"✓ Found {num_regions} diseased region(s)")
    
    if output_path:
        cv2.imwrite(output_path, highlighted)
        print(f"✓ Result saved to: {output_path}")
    
    return highlighted

if __name__ == '__main__':
    import sys
    
    # Check if models exist
    resnet_path = 'best_resnet50_plant_model.pth'
    unet_path = 'best_unet_model.pth'
    
    if not Path(resnet_path).exists():
        print(f"Error: ResNet50 model not found at {resnet_path}")
        print("Please train the ResNet50 model first using main.py")
        sys.exit(1)
    
    if not Path(unet_path).exists():
        print(f"Error: U-Net model not found at {unet_path}")
        print("Please train the U-Net model first using train_unet.py")
        sys.exit(1)
    
    # Load models
    resnet_model, unet_model = load_models(resnet_path, unet_path)
    
    # Example: Process a test image
    # You can modify this to process your own images
    dataset_path = Path(r"C:\Users\wongj\OneDrive\Desktop\github\CS455\Final\PlantVillage")
    
    # Try to find a test image
    test_images = []
    diseased_dir = dataset_path / 'Pepper__bell___Bacterial_spot'
    healthy_dir = dataset_path / 'Pepper__bell___healthy'
    
    if diseased_dir.exists():
        test_images.extend(list(diseased_dir.glob('*.JPG'))[:2])
    if healthy_dir.exists():
        test_images.extend(list(healthy_dir.glob('*.JPG'))[:1])
    
    if not test_images:
        print("\nNo test images found. Please provide an image path:")
        image_path = input("Enter image path: ")
        if Path(image_path).exists():
            output_path = str(Path(image_path).stem) + '_result.jpg'
            process_image(resnet_model, unet_model, image_path, output_path)
        else:
            print("Image not found!")
    else:
        print(f"\nFound {len(test_images)} test image(s). Processing...")
        for i, img_path in enumerate(test_images):
            output_path = f'test_result_{i+1}.jpg'
            process_image(resnet_model, unet_model, str(img_path), output_path)
        
        print(f"\n{'='*60}")
        print("Demo complete! Check the output images.")
        print(f"{'='*60}")

