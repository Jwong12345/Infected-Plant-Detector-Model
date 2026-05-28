import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset path
DATASET_PATH = r"C:\Users\wongj\OneDrive\Desktop\github\CS455\Final\PlantVillage"

# Define classes - only pepper bell classes for testing
CLASSES = {
    'Pepper__bell___Bacterial_spot': 0,  # Diseased
    'Pepper__bell___healthy': 1  # Healthy
}

class PlantDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from specified classes only
        for class_name, label in classes.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                # Get all image files
                image_files = []
                for ext in ['*.JPG', '*.jpg', '*.png', '*.jpeg']:
                    image_files.extend(list(class_dir.glob(ext)))
                
                for img_path in image_files:
                    self.images.append(str(img_path))
                    self.labels.append(label)
                
                print(f"Loaded {len(image_files)} images from {class_name} (label: {label})")
        
        print(f"Total images loaded: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
print("Loading dataset...")
full_dataset = PlantDataset(DATASET_PATH, CLASSES, transform=None)

# Split dataset into train and validation (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices = random_split(
    range(len(full_dataset)), 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Create train and validation datasets with appropriate transforms
train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)

# Apply transforms by creating wrapper
class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.subset)

train_dataset = TransformDataset(train_dataset, train_transform)
val_dataset = TransformDataset(val_dataset, val_transform)

# Data loaders
batch_size = 32
# Use num_workers=0 on Windows to avoid multiprocessing issues
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Load ResNet50 model
print("Loading ResNet50 model...")
model = models.resnet50(weights='IMAGENET1K_V2')
num_features = model.fc.in_features

# Replace the final layer for binary classification (healthy vs diseased)
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # Training loop
    num_epochs = 5  # Reduced for quick testing
    best_val_acc = 0.0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet50_plant_model.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
    
    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Model saved as: best_resnet50_plant_model.pth')
