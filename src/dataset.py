import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

class DDataset(Dataset):
    def __init__(self, img_dir, files, labels=None):
        self.labels = labels
        self.img_dir = img_dir
        self.files = files
        self.cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.cache:
            img = self.cache[idx]
        else:
            img_name = os.path.join(self.img_dir, str(self.files[idx])) + ".jpg"
            img = Image.open(img_name)
            if img.mode != 'RGB':
                img.convert('RGB')
            if min(img.size) > 560:
                img = resize(img, 560)
            self.cache[idx] = img

        if self.labels is None:
            lab = -1
        else:
            lab = self.labels[idx]
        return img, lab

