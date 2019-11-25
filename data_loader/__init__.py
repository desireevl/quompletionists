import torch
from torchvision import datasets, transforms

train_fashion_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        '/tmp', train=True, download=True,
        transform=transforms.Compose(
            [transforms.ToTensor()]
        )
    )
)

if __name__ == '__main__':
    for batch_idx, (data, target) in enumerate(train_fashion_loader):
        print(batch_idx)