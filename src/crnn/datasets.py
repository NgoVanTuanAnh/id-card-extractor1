import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class IDDataset(Dataset):
    def __init__(self, config, data, transform=False):
        self.data = data
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, gt = self.data.iloc[idx]
        image = Image.open(os.path.join(self.config['data_dir'], path))
        transform = transforms.Compose([
            transforms.Pad(padding=(5, 0, 5, 0), fill=200),
            transforms.ToTensor(),
            transforms.Resize(size=(self.config['img_height'], self.config['img_width'])),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if self.transform:
            trans = self.transform.transforms
            for i, tran in enumerate(trans):
                transform.transforms.insert(i+1, tran)
                
        image = transform(image)
        return image, gt