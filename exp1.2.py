import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.image_paths = []
        self.labels = []

        breed_dirs = [d for d in os.listdir(os.path.join(root_dir, "images/Images"))
                     if os.path.isdir(os.path.join(root_dir, "images/Images", d))]

        self.class_to_idx = {breed: idx for idx, breed in enumerate(sorted(breed_dirs))}
        self.classes = list(self.class_to_idx.keys())

        for breed in breed_dirs:
            breed_path = os.path.join(root_dir, "images/Images", breed)
            for fname in os.listdir(breed_path):
                if fname.endswith('.jpg'):
                    img_path = os.path.join(breed_path, fname)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[breed])
                    except (IOError, OSError) as e:
                        print(f"Removed corrupt file {img_path}: {str(e)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(180, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

full_dataset = StanfordDogsDataset(
    root_dir="stanforddogsarchive",
    transform=transform
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

assert len(full_dataset.classes) == 120, "Should have 120 dog breeds"

class DogsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128*20*20, 512), nn.ReLU(),
            nn.Linear(512, 120)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = DogsModel().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

def train(epochs=30):
    best_acc = 0.0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = evaluate(val_loader)
        print(f'Epoch {epoch+1:02d}')
        print(f'Train Loss: {total_loss/len(train_loader):.4f} | Acc: {100*correct/len(train_dataset):.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'stanford_dogs.pth')

    print(f'Best Validation Accuracy: {best_acc:.2f}%')

def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
    return 100 * correct / len(loader.dataset)

if __name__ == '__main__':
    train(epochs=30)
