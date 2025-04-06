import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# --------------------- CUDA Configuration ---------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# --------------------- Transfer Model Architecture ---------------------
class TransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Modified feature extractor with padding
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Added padding
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Added padding
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Added padding
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Updated classifier with correct input size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 22 * 22, 512),  # Corrected input size
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze()

# --------------------- Dataset Class (Same as Exp1.1) ---------------------
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Cat', 'Dog']
        self.samples = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    self.samples.append((fpath, class_idx))
                except (IOError, OSError) as e:
                    print(f"Removing corrupt file {fpath}: {str(e)}")
                    os.remove(fpath)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# --------------------- Data Preparation ---------------------
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(180, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# Load Cats vs Dogs dataset
full_dataset = CatDogDataset(
    root_dir="kagglecatsanddogs_3367a/PetImages",
    transform=transform
)

# Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(1337)
)

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, 
    num_workers=4, pin_memory=True
)

# --------------------- Model Initialization ---------------------
model = TransferModel().to(device)

# Load pretrained weights (excluding final layer)
# Load pretrained weights
pretrained_dict = torch.load('stanford_dogs.pth')
model_dict = model.state_dict()

# 1. Filter out unnecessary keys (final layer)
pretrained_dict = {
    k: v for k, v in pretrained_dict.items() 
    if k in model_dict and v.size() == model_dict[k].size()
}

# 2. Overwrite entries in the existing state dict
model_dict.update(pretrained_dict)

# 3. Load the new state dict
model.load_state_dict(model_dict)

print("\nSuccessfully loaded pretrained weights:")
for k in pretrained_dict:
    print(f"- Loaded: {k}")
# --------------------- Training Setup ---------------------
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

# --------------------- Enhanced Training Loop ---------------------
def train(epochs):
    best_val_acc = 0.0
    print("\nStarting transfer learning...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Model architecture:")
    print(model)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        # Test feature extractor output shape
        test_input = torch.randn(1, 3, 180, 180).to(device)
        with torch.no_grad():
            features = model.features(test_input)
            print("Feature extractor output shape:", features.shape)
            classifier_input = model.classifier[0](features)
            print("Classifier input size:", classifier_input.shape[1])
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        
        # Calculate metrics
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        print(f"\nEpoch {epoch+1:02d}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'transfer_model.pth')
            print("★ New best model saved!")
    
    print(f"\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# --------------------- Execution ---------------------
if __name__ == '__main__':
    train(epochs=50)
    torch.save(model.state_dict(), 'final_transfer_model.pth')