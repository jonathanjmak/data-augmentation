import os
import pickle
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm, trange

class StylizedDataset(VisionDataset):
    def __init__(self, p_monet, p_udnie):
        super(StylizedDataset, self).__init__(root="./data")
        self.data, self.labels = [], []
        for artist in ["monet", "udnie"]:
            for i in range(10):
                with open(f"../style/datasets/{artist}_data_{i}.pkl", "rb") as f:
                    data_for_label = pickle.load(f)

                num = int((p_monet if artist == "monet" else p_udnie) * len(data_for_label))
                data_for_label = data_for_label[:num]

                labels = i * torch.ones(len(data_for_label))

                if not len(self.data):
                    self.data = data_for_label
                    self.labels = labels
                else:
                    self.data = torch.cat((self.data, data_for_label), 0)
                    self.labels = torch.cat((self.labels, labels), 0)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class ThresholdedDataset(VisionDataset):
    def __init__(self, threshold):
        super(ThresholdedDataset, self).__init__(root="./data")
        self.data = []
        self.labels = []

        if threshold is None:
            return

        assert os.path.exists(f"../cgan/thresholded_datasets/thresh_{threshold:.2f}/")

        for i in range(10):
            with open(f"../cgan/thresholded_datasets/thresh_{threshold:.2f}/data_{i}.pkl", "rb") as f:
                data_for_label = pickle.load(f)

            labels = i * torch.ones(len(data_for_label))

            if not len(self.data):
                self.data = data_for_label
                self.labels = labels
            else:
                self.data = torch.cat((self.data, data_for_label), 0)
                self.labels = torch.cat((self.labels, labels), 0)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class CombinedDataset(VisionDataset):
    def __init__(self, p_cifar, p_thresholded, p_monet, p_udnie, device=None, transform=None, p_val=0.0, train=True, threshold=None):
        super(CombinedDataset, self).__init__(root="./data")

        self.transform = transform
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert p_val <= 1.0 and p_val >= 0.
        assert p_val or train
        assert p_cifar <= 1.0 and p_cifar >= 0.
        assert p_thresholded <= 1.0 and p_thresholded >= 0.
        assert (threshold is None) == (p_thresholded == 0.)
        assert p_monet <= 1.0 and p_monet >= 0.
        assert p_udnie <= 1.0 and p_udnie >= 0.

        cifar = datasets.CIFAR10(
            root='./data', 
            train=train, 
            download=True,
            transform=transforms.ToTensor(),
            target_transform=lambda x: torch.tensor(int(x)),
        )
        thresholded = ThresholdedDataset(threshold=threshold)
        stylized = StylizedDataset(p_monet=p_monet,p_udnie=p_udnie)

        self.data = []

        n_samples_from_cifar = int(p_cifar * len(cifar))
        n_samples_from_thresholded = int(p_thresholded * len(thresholded))

        np.random.seed(0)
        choices_from_cifar = np.random.choice(a=len(cifar), size=n_samples_from_cifar, replace=False)
        choices_from_thresholded = np.random.choice(a=len(thresholded), size=n_samples_from_thresholded, replace=False)

        if p_val:
            n_train_thresholded = int((1 - p_val) * n_samples_from_thresholded)
            if train:
                choices_from_thresholded = choices_from_thresholded[:n_train_thresholded]
            else:
                choices_from_thresholded = choices_from_thresholded[n_train_thresholded:]

        for i in choices_from_cifar:
            x, y = cifar[i]
            x = x.to(self.device)
            y = y.to(self.device)
            self.data.append((x, y, 1.))

        for i in choices_from_thresholded:
            x, y = thresholded[i]
            x = x.to(self.device)
            y = y.to(self.device)
            self.data.append((x, y, 0.))

        for x, y in stylized:
          x = x.to(self.device)
          y = y.to(self.device)
          self.data.append((x, y, 0.))

        print(len(self.data))
        assert len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1].long()
        real = self.data[index][2]
        if self.transform:
            x = self.transform(x)
        x = x.detach()
        y = y.detach()
        return (x, y, real)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dss = [
        StylizedDataset(1.0, 1.0),
        StylizedDataset(0.5, 0.5),
        CombinedDataset(p_cifar=1., p_thresholded=1., threshold=0.9, p_monet=1., p_udnie=1.),
        ThresholdedDataset(0.90),
        # ThresholdedDataset(0.75),
        ThresholdedDataset(0.50),
        # ThresholdedDataset(0.25),
        ThresholdedDataset(0.10),
        ThresholdedDataset(0.00),
    ]
    for ds in dss:
        for img in tqdm(ds):
            pass
        print()