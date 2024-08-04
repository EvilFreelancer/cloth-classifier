import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from cloth_classifier.preprocess_image import get_transformation


def prepare_dataset(batch_size: int = 64, img_weight=28, img_height=28):
    # Get transformation
    transform = get_transformation(img_weight, img_height)

    # Download training and testing data
    train_ds = datasets.FashionMNIST('F_MNIST_data', download=True, train=True, transform=transform)
    test_ds = datasets.FashionMNIST('F_MNIST_data', download=True, train=False, transform=transform)
    print({"len(train_ds)": len(train_ds), "len(test_ds)": len(test_ds)})

    # Split train set into training (80%) and validation set (20%)
    train_num = len(train_ds)
    indices = list(range(train_num))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * train_num))
    val_idx, train_idx = indices[:split], indices[split:]
    print({"len(train_idx)": len(train_idx), "len(val_idx)": len(val_idx)})

    # Prepare dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_dl = DataLoader(train_ds, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_dl, val_dl, test_dl


if __name__ == '__main__':
    train_dl, val_dl, test_dl = prepare_dataset()
