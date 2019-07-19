import os
import pandas as pd
from tqdm import tqdm
from src.dataset import DDataset
from src.augmentator import Augmentator

ds = pd.read_csv("images/train.csv")
train = DDataset('images/train', files=ds['fileName'], labels=ds['tag'])
ag_train = Augmentator(train, mode="test")

repaired = 0
for i in tqdm(range(len(train))):
    try:
        img, lab = ag_train[i]
    except:
        img_name = os.path.join(train.img_dir, str(train.files[i])) + ".jpg"
        os.system("convert %s tmp.png" % img_name)
        os.system("convert tmp.png %s" % img_name)
        repaired += 1
print("%d train images repaired" % repaired)


ts_files = list(range(9738))
ts_files.remove(1213)
ts_files.remove(3574)
ts_files.remove(6086)
test = DDataset('images/test', files=ts_files)
ag_test = Augmentator(test, mode="test")

repaired = 0
for i in tqdm(range(len(test))):
    try:
        img, lab = ag_test[i]
    except:
        img_name = os.path.join(test.img_dir, str(test.files[i])) + ".jpg"
        os.system("convert %s tmp.png" % img_name)
        os.system("convert tmp.png %s" % img_name)
        repaired += 1
print("%d test images repaired" % repaired)

