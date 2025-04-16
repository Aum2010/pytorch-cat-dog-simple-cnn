import torch
from torch.utils.data import DataLoader
from glob import glob
from dataset import FolderLabelImageDataset
from model import SimpleCNN
from torchvision import transforms

# Device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

# Load test dataset
img_paths = glob("test/*/*.jpg")
test_dataset = FolderLabelImageDataset(img_paths , transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Evaluation loop
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
