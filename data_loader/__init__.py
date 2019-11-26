import torch
import os
import random

from matplotlib import pyplot as plt
from torchvision import datasets, transforms


class DataTransformer(object):
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        data = next(self.dataloader)
        label = None

        try:
            label = data[1]
        except:
            pass

        # Calculate x, y for cropping
        source = data[0].clone()
        source = source.squeeze()

        original = data[0].clone()
        w, h = (3, 3) # output size
        _, _, tw, th = original.size()

        y = random.randint(0, th - h)
        x = random.randint(0, tw - w)

        # Remove 1 dim data
        original = original.squeeze()

        # Crop
        original = original[y:y+h, x:x+w]

        original = original.reshape(-1)
        target = original[-1].clone()
        original[-1] = 0.0

        return (2, 2), original, target, label, source, (x, y)


file_abs_dir_path = os.path.dirname(os.path.realpath(__file__))

celeb_dataset_path = os.path.join(
    os.path.join(os.path.join(
        file_abs_dir_path,
        '..'), 'data'), '5-celeb-faces'
)
train_celeb_dataset_path = os.path.join(celeb_dataset_path, 'train')
test_celeb_dataset_path = os.path.join(celeb_dataset_path, 'test')

transformations = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor()
])

train_fashion_loader = DataTransformer(
    torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            '/tmp', train=True, download=True,
            transform=transformations
        )
    )
)

test_fashion_loader = DataTransformer(
    torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            '/tmp', train=False, download=True,
            transform=transformations
        )
    )
)

train_celeb_face_loader = DataTransformer(
    torch.utils.data.DataLoader(
        datasets.ImageFolder(
            train_celeb_dataset_path,
            transform=transformations
        )
    )
)

test_celeb_face_loader = DataTransformer(
    torch.utils.data.DataLoader(
        datasets.ImageFolder(
            test_celeb_dataset_path,
            transform=transformations
        )
    )
)


if __name__ == '__main__':
    for idx, ((i, j), X, Y, label, source, (cropX, cropY)) in enumerate(train_fashion_loader):
        print(idx, (i, j), X, Y, label)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(source, cmap=plt.cm.gray)
        plt.show()
