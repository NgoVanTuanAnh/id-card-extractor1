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
        image = Image.open(os.path.join(self.config.CRNN.TRAINING.DATA_DIR, path))
        transform = transforms.Compose([
            transforms.Pad(padding=self.config.CRNN.PAD.PADDING, fill=self.config.CRNN.PAD.FILL),
            transforms.ToTensor(),
            transforms.Resize(size=(self.config.CRNN.HEIGHT, self.config.CRNN.WIDTH)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if self.transform:
            trans = self.transform.transforms
            for i, tran in enumerate(trans):
                transform.transforms.insert(i+1, tran)
                
        image = transform(image)
        return image, gt