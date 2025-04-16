import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FolderLabelImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        # self.transform = transform or transforms.Compose([
        #     transforms.Resize((64, 64)),
        #     transforms.ToTensor()
        # ])
        self.transform = transform
        self.class_map = {"cats": 0, "dogs": 1}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label_str = os.path.basename(os.path.dirname(img_path))
        label = self.class_map[label_str]

        if self.transform:
            image = self.transform(image)
            
        return image, label
