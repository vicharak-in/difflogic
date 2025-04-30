import os
from typing import Any, Callable, Optional, Tuple, Dict

import torch
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class CIFAR10_3Thresholds(VisionDataset):
    """
    CIFAR-10 dataset with 3-threshold binarization preprocessing.
    
    Args:
        root (str): Root directory of dataset.
        train (bool): Load training data if True, else test data.
        transform (callable, optional): Extra transform to apply after thresholding.
        target_transform (callable, optional): Target transform.
        download (bool): Download dataset if not found.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ) -> None:
        # Define threshold transformation
        def threshold_transform(x):
            x = torchvision.transforms.ToTensor()(x)
            return torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0)

        final_transform = threshold_transform if transform is None else \
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(threshold_transform),
                transform
            ])

        super(CIFAR10_3Thresholds, self).__init__(
            root, transform=final_transform, target_transform=target_transform
        )

        self.base = CIFAR10(
            root=root,
            train=train,
            download=download
        )

        self.data = self.base.data
        self.targets = self.base.targets
        self.classes = self.base.classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        return f"Split: {'Train' if self.base.train else 'Test'}"


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = CIFAR10_3Thresholds('data-cifar', train=True, download=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True)

    imgs, labels = next(iter(loader))
    fig, axs = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axs[i].imshow(imgs[i].view(3, 32, 32)[0].numpy(), cmap='gray')
        axs[i].set_title(f"{ds.classes[labels[i]]}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

