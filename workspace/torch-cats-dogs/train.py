import torch
from torch.utils.data import DataLoader
from glob import glob
from dataset import FolderLabelImageDataset
from model import SimpleCNN
from logs import print_stats
from torchvision import transforms

# Cats and Dogs Dataset
# https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification

# Settings
batch_size = 64
epochs = 10
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

# Dataset
img_paths = glob("train/*/*.jpg")
dataset = FolderLabelImageDataset(img_paths , transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = SimpleCNN().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train loop
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print_stats(epoch, running_loss / len(dataloader), accuracy)

print("Training done. Saving model...")
torch.save(model.state_dict(), "model.pth")