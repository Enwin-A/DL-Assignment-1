import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

class Experiment3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 20 * 20, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze()

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

transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(180, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

full_dataset = CatDogDataset(
    root_dir="kagglecatsanddogs_3367a/PetImages",
    transform=transform
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(1337)
)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32,
    num_workers=4, pin_memory=True
)

model = Experiment3Model().to(device)

pretrained_dict = torch.load('stanford_dogs.pth')
model_dict = model.state_dict()

keys_to_load = [
    'features.6.weight',
    'features.6.bias'
]
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                  if k in keys_to_load and k in model_dict}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)

for name, param in model.named_parameters():
    if not (name.startswith('features.0') or
           name.startswith('features.3') or
           name.startswith('classifier.4')):
        param.requires_grad = False

optimizer = optim.RMSprop([
    {'params': model.features[0].parameters()},
    {'params': model.features[3].parameters()},
    {'params': model.classifier[4].parameters()}
], lr=0.0001)

criterion = nn.BCELoss()

def train(epochs):
    best_val_acc = 0.0
    print("\nStarting Experiment 3 training...")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
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

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        test_input = torch.randn(1, 3, 180, 180).to(device)
        with torch.no_grad():
            features = model.features(test_input)
            print("Feature shape:", features.shape)
            print("Classifier input:", model.classifier[0](features).shape[1])

            test_input = torch.randn(1, 3, 180, 180).to(device)
            features = model.features(test_input)
            if epoch == 0:
                print("\nFeature verification:")
                print(f"Feature extractor output shape: {features.shape}")
                print(f"Classifier input size: {model.classifier[0](features).shape[1]}\n")

            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val

        print(f"\nEpoch {epoch+1:02d}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'exp3_best_model.pth')
            print("★ New best model saved!")

    print(f"\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    train(epochs=50)
    torch.save(model.state_dict(), 'exp3_final_model.pth')
