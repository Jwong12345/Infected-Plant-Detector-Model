import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import argparse
from unet_model import UNet

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing for ResNet50
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Image preprocessing for U-Net (no normalization needed for visualization)
unet_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_resnet_model(model_path, device):
    """Load trained ResNet50 model"""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_unet_model(model_path, device):
    """Load trained U-Net model"""
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def classify_image(resnet_model, image_path, device):
    """Classify image as healthy or diseased using ResNet50"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Preprocess for ResNet50
    input_tensor = resnet_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = resnet_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_names = ['Diseased', 'Healthy']
    return predicted_class == 0, confidence, class_names[predicted_class], original_size

def segment_image(unet_model, image_path, device):
    """Segment diseased regions using U-Net"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Preprocess for U-Net
    input_tensor = unet_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mask = unet_model(input_tensor)
        mask = mask.squeeze().cpu().numpy()
    
    # Resize mask to original image size
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
    
    return mask_resized, image

def highlight_diseased_regions(image, mask, threshold=0.5):
    """Use OpenCV to highlight diseased regions on the image"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Create binary mask
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create overlay image
    overlay = img_array.copy()
    highlighted = img_array.copy()
    
    # Draw filled contours with transparency
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)  # Red fill
    highlighted = cv2.addWeighted(overlay, 0.4, highlighted, 0.6, 0)
    
    # Draw contour outlines
    cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)  # Red outline
    
    # Add text label
    if len(contours) > 0:
        cv2.putText(highlighted, 'Diseased Region', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return highlighted, binary_mask, len(contours)

def process_image(resnet_model, unet_model, image_path, output_path=None, device=device):
    """Complete pipeline: Classify -> Segment -> Visualize"""
    print(f"\nProcessing: {image_path}")
    
    # Step 1: Classify with ResNet50
    is_diseased, confidence, class_name, original_size = classify_image(
        resnet_model, image_path, device
    )
    
    print(f"Classification: {class_name} (confidence: {confidence:.2%})")
    
    if not is_diseased:
        print("Image is healthy - no segmentation needed.")
        # Still show the image with classification label
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.putText(img_array, f'Healthy ({confidence:.1%})', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, img_array)
            print(f"Result saved to: {output_path}")
        
        return img_array, None
    
    # Step 2: Segment with U-Net
    print("Image is diseased - performing segmentation...")
    mask, image = segment_image(unet_model, image_path, device)
    
    # Step 3: Highlight with OpenCV
    highlighted, binary_mask, num_regions = highlight_diseased_regions(image, mask)
    
    # Add classification info to image
    cv2.putText(highlighted, f'{class_name} ({confidence:.1%})', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(highlighted, f'Regions: {num_regions}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    print(f"Found {num_regions} diseased region(s)")
    
    if output_path:
        cv2.imwrite(output_path, highlighted)
        print(f"Result saved to: {output_path}")
    
    return highlighted, binary_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plant Disease Detection and Segmentation')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--resnet_model', type=str, default='best_resnet50_plant_model.pth',
                       help='Path to ResNet50 model')
    parser.add_argument('--unet_model', type=str, default='best_unet_model.pth',
                       help='Path to U-Net model')
    parser.add_argument('--output', type=str, default=None, help='Path to save output image')
    
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    resnet_model = load_resnet_model(args.resnet_model, device)
    unet_model = load_unet_model(args.unet_model, device)
    print("Models loaded successfully!")
    
    # Process image
    if args.output is None:
        args.output = str(Path(args.image).stem) + '_result.jpg'
    
    result, mask = process_image(resnet_model, unet_model, args.image, args.output, device)
    
    print("\nProcessing complete!")

