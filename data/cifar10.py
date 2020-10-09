import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl


class Cifar10DataModule(pl.LightningDataModule):

    def __init__(self, dataset, data_path, batch_size, num_workers, **kw):
        super().__init__()
        self.dataset = dataset.lower()
        assert self.dataset == 'cifar10', 'The dataset should be cifar10.'
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_classes = 10
        self.dims = (3, 32, 32)


      # When doing distributed training, Datamodules have two optional arguments for
      # granular control over download/prepare/splitting data:

      # OPTIONAL, called only on 1 GPU/machine
    #   def prepare_data(self):
    #       datasets.CIFAR10(self.datapath, train=True, download=True)
    #       datasets.CIFAR10(self.datapath, train=False, download=True)

      # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
          # transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        if stage == 'fit' or stage is None:
            self.train_set = torchvision.datasets.CIFAR10(
                root=self.data_path, train=True, download=True, transform=transform_train)

            self.val_set = torchvision.datasets.CIFAR10(
                root=self.data_path, train=False, download=True, transform=transform_test)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_set = torchvision.datasets.CIFAR10(
                root=self.data_path, train=False, download=True, transform=transform_test)
        

      # return the dataloader for each split
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_datalodaer = torch.utils.data.DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return val_datalodaer

    def test_dataloader(self):
        test_datalodaer = torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return test_datalodaer

