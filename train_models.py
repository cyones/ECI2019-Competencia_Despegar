import torch as tr
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import CyclicLR

from src.dataset import DDataset
from src.sampler import DSampler
from src.model import Model
from src.augmentator import Augmentator

tr.backends.cudnn.deterministic = True
tr.backends.cudnn.benchmark = False

batch_size = 24
n_classes = 16

ds = pd.read_csv("images/train.csv")
train = DDataset('images/train', files=ds['fileName'], labels=ds['tag'])

logger = open("logfiles/trainer0.log", 'w', buffering=1)

dev = tr.device("cuda:0")

nmodel = 0

rskf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
for train_idx, valid_idx in rskf.split(ds['tag'], y=ds['tag']):
    if os.path.isfile("models/%d.pmt" % nmodel):
        print("Model %d already trained" % nmodel)
        nmodel += 1
        continue
    tr.cuda.empty_cache()
    tr.manual_seed(nmodel)
    np.random.seed(nmodel)
    ftrain = Augmentator(train, mode='train', indices=train_idx)
    fvalid = Augmentator(train, mode='valid', indices=valid_idx)

    model = Model(n_classes).to(dev)
    model.freeze_resnet()
    criterion = nn.NLLLoss()
    optimizer = tr.optim.Adam(model.parameters())
    optimizer = tr.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0)
    lr_scheduler = CyclicLR(optimizer, 0.0001, 0.1, step_size_up=1024,
                            mode="exp_range", gamma=0.9, base_momentum=0.5)

    train_sampler = DSampler(list(ds['tag'][train_idx]), len(ftrain))
    train_loader = DataLoader(ftrain, batch_size=batch_size, sampler=train_sampler,
            num_workers=8, pin_memory=True)
    valid_loader = DataLoader(fvalid, batch_size=batch_size, num_workers=8,
           pin_memory=True)

    train_loss = 1
    train_acc = 0
    valid_acc = 0
    best_valid_acc = 0
    early_stop = 0
    epoch = 0
    while early_stop < 16:
        if epoch == 4:
            tr.cuda.empty_cache()
            model.unfreeze_resnet()
        ib = 0
        model = model.train()
        for img, lbs in train_loader:
            optimizer.zero_grad()
            pred = model(img.to(dev))
            loss = criterion(pred, lbs.to(dev))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            pred = tr.max(pred, 1)[1].cpu().detach().numpy()
            new_train_acc = balanced_accuracy_score(lbs.numpy(), pred)
            train_acc = 0.1 * new_train_acc + 0.9 * train_acc
            train_loss = 0.1 * loss.item() + 0.9 * train_loss
            print('Model: %d, Epoch: %d, batch: %d, loss: %.4f, acc: %.4f' %
                    (nmodel, epoch, ib, train_loss, train_acc), end='\r')
            ib += 1
        del img, lbs, loss, pred
        print('Model: %d, Epoch: %d, batch: %d, loss: %.4f, acc: %.4f' %
                (nmodel, epoch, ib, train_loss, train_acc), end=', ')

        model = model.eval()
        pred = tr.LongTensor([])
        labels = tr.LongTensor([])
        for img, lbs in valid_loader:
            lpred = tr.zeros(img.shape[0], 16)
            for r in range(4):
                lpred += tr.exp(model(img.to(dev)).detach().cpu()) / 4
            pred = tr.cat((pred, tr.max(lpred, 1)[1]))
            labels = tr.cat((labels, lbs))
        valid_acc = balanced_accuracy_score(labels.numpy(), pred.detach().numpy())
        print('Acc: %.4f' % (valid_acc), end=', ')
        logger.write('%d, %d, %.4f, %.4f, %.4f' %
                (nmodel, epoch, train_loss, train_acc, valid_acc))
        del img, lbs, labels, pred

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            early_stop = 0
            tr.save(model.state_dict(), 'models/%d.pmt' % nmodel)
            print('Improvement')
        else:
            early_stop += 1
            print('No improvement')
        epoch += 1
    nmodel += 1

