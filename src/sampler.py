import torch as tr
from torch.utils.data.sampler import Sampler

class DSampler(Sampler):
    def __init__(self, labels, nsamples):
        self.nsamples =  nsamples
        label_to_count = {}
        for idx in range(len(labels)):
            if labels[idx] in label_to_count:
                label_to_count[labels[idx]] += 1
            else:
                label_to_count[labels[idx]] = 1

        weights = [1.0 / label_to_count[labels[idx]]
                   for idx in range(len(labels))]
        self.weights = tr.DoubleTensor(weights)

    def __iter__(self):
        return iter(tr.multinomial(self.weights, self.nsamples, replacement=True))

    def __len__(self):
        return self.nsamples
