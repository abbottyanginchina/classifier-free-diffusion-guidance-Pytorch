from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.utils.data import Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def load_data(batchsize:int, numworkers:int):
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root = './data/cifar-10',
                        train = True,
                        download = True,
                        transform = trans
                    )
    
    # processing training data
    idxs = np.load('./data/idxs.npz')['arr_0']
    data_train = DatasetSplit(data_train, idxs)

    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        sampler = sampler,
                        drop_last = True
                    )
    return trainloader, sampler

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5
