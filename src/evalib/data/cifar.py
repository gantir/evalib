import torch
from torch import cuda
from torchvision import datasets, transforms
from troch.utils.data import DataLoader


class CIFAR:
    def __init__(self, args):
        super(CIFAR, self).__init__()
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        self._args = args

        self._prepare_dataloader()

    def _train_mean(self):
        return (0.4914, 0.4822, 0.4465)

    def _train_std(self):
        return (0.2023, 0.1994, 0.2010)

    def _test_mean(self):
        return (0.4914, 0.4822, 0.4465)

    def _test_std(self):
        return (0.2023, 0.1994, 0.2010)

    def _prepare_data(self):
        # Train Phase transformations
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self._train_mean(), self._train_std()),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transforms
        )

        # Test Phase transformations
        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self._test_mean(), self._test_std()),
            ]
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=test_transforms
        )

        return train_dataset, test_dataset

    def _prepare_dataloader(self):
        train_dataset, test_dataset = self._prepare_data()

        torch.manual_seed(self._args.SEED)
        dlargs = {"batch_size": self._args.batch_size}
        if cuda.is_available():
            torch.cuda.manual_seed(self._args.SEED)
            dlargs = {
                "num_workers": self._args.num_workers,
                "pin_memory": True,
                "batch_size": self._args.batch_size_cuda,
            }

        self.train_loader = DataLoader(train_dataset, **dlargs)
        self.test_loader = DataLoader(test_dataset, **dlargs)
