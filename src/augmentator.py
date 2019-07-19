from torch.utils.data import Dataset
import torchvision.transforms as trn

class Augmentator(Dataset):
    def __init__(self, dataset, mode, indices=None):
        self.dataset = dataset
        if indices is None: self.indices = list(range(len(dataset)))
        else: self.indices = indices
        self.mode = mode

    def __getitem__(self, idx):
        img, lbl = self.dataset[self.indices[idx]]
        if self.mode=='train':
            img = self.train_trans(img)
        if self.mode=='valid':
            img = self.valid_trans(img)
        if self.mode=='test':
            img = self.test_trans(img)
        return img, lbl

    def __len__(self):
        return len(self.indices)

    train_aug = trn.Compose([
        trn.RandomAffine(15),
        trn.RandomResizedCrop(280, scale=(0.5, 1.5), ratio=(0.9,1.1)),
        trn.RandomHorizontalFlip(),
        trn.RandomGrayscale(0.2),
        trn.ColorJitter(0.25, 0.25, 0.25)])

    valid_aug = trn.Compose([
        trn.RandomResizedCrop(280, scale=(1.0, 1.0), ratio=(1.0,1.0)),
        trn.RandomGrayscale(0.2),
        trn.RandomHorizontalFlip()])

    test_aug = trn.Compose([
        trn.RandomResizedCrop(280, scale=(1.0, 1.0), ratio=(1.0,1.0)),
        trn.RandomGrayscale(0.2),
        trn.RandomHorizontalFlip()])

    tensorize = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225])])

    train_trans = trn.Compose([train_aug, tensorize])
    valid_trans = trn.Compose([valid_aug, tensorize])
    test_trans = trn.Compose([test_aug, tensorize])
