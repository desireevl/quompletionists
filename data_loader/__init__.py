import torch
import os

from torchvision import datasets, transforms


file_abs_dir_path = os.path.dirname(os.path.realpath(__file__))

celeb_dataset_path = os.path.join(
    os.path.join(os.path.join(
        file_abs_dir_path,
        '..'), 'data'), '5-celeb-faces'
)
train_celeb_dataset_path = os.path.join(celeb_dataset_path, 'train')
test_celeb_dataset_path = os.path.join(celeb_dataset_path, 'test')

transformations = transforms.Compose([
    transforms.RandomCrop((3, 3)),
    transforms.ToTensor()
])

train_fashion_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        '/tmp', train=True, download=True,
        transform=transformations
    )
)

test_fashion_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        '/tmp', train=False, download=True,
        transform=transformations
    )
)

train_celeb_face_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        train_celeb_dataset_path,
        transform=transformations
    )
)

test_celeb_face_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        test_celeb_dataset_path,
        transform=transformations
    )
)

if __name__ == '__main__':
    for batch_idx, (data, target) in enumerate(train_celeb_face_loader):
        print(batch_idx, data.data.numpy())
