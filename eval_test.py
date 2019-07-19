import torch as tr
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import Subset, DataLoader

from src.dataset import DDataset
from src.sampler import DSampler
from src.model import Model
from src.augmentator import Augmentator

batch_size = 24
n_classes = 16

ids = list(range(9739))
ids.remove(1213)
ids.remove(3574)
ids.remove(6086)
test = Augmentator(DDataset('images/test', files=ids), mode='test')
test_loader = DataLoader(test, batch_size=batch_size, num_workers=8, pin_memory=True)

dev = tr.device("cuda:0")


test_preds = tr.zeros(len(test), n_classes)
for nmodel in range(10):
    print("Evaluating with model %d: [" % nmodel, end='')
    tr.cuda.empty_cache()
    tr.manual_seed(nmodel)
    np.random.seed(nmodel)

    model = Model(n_classes, pretrained=False)
    model.load_state_dict(tr.load('models/%d.pmt' % nmodel))
    model = model.to(dev).eval()
    for r in range(3):
        print("|", end='')
        pred = tr.Tensor(len(test), n_classes)
        ib = 0
        for img, _ in test_loader:
            pred[ib:(ib+batch_size)] = model(img.to(dev)).detach().cpu()
            ib += batch_size
        test_preds += tr.exp(pred) / (3 * 10)
    print("]")
idx_pred = np.argmax(test_preds, axis=1)

pd.DataFrame({'id' : ids, 'target' : idx_pred}).to_csv("preds/submission.csv",
                                                        index=False,
                                                        header=False)
