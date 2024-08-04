import torch
from torchvision import transforms
from PIL import Image


def get_transformation(weight=28, height=28):
    return transforms.Compose([
        transforms.Resize((weight, height)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def preprocess_image(image: Image.Image, weight=28, height=28) -> torch.Tensor:
    transform = get_transformation(weight, height)
    return transform(image)
