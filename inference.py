import torch
import random
import matplotlib.pyplot as plt

from cloth_classifier.prepare_dataset import prepare_dataset
from cloth_classifier.models import AdvancedNet, SimpleNet, RhombusNet

CLASSES = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]


def inference(model, img):
    # Calculate the class probabilities (softmax) for img
    prediction = model(img)
    proba = torch.exp(prediction)
    return proba


def prepare_image(dataloader, index=49):
    # Test out the network!
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    img, label = images[index], labels[index]

    # Convert 2D image to 1D vector
    img_vec = img.view(img.shape[0], -1)

    # Original image for preview
    img_orig = images[index].cpu().numpy().squeeze()
    return img_orig, img_vec, label


def explain(image, label, proba):
    # Plot the image and probabilities
    fig, (ax1, ax2) = plt.subplots(figsize=(13, 6), nrows=1, ncols=2)
    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title(CLASSES[label.item()])
    ax2.bar(range(10), proba.detach().cpu().numpy().squeeze())
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(CLASSES, size='small')
    ax2.set_title('Predicted Probabilities')
    plt.savefig('explain.png')


def format(prediction):
    # Calculate the class probabilities (softmax) for img
    proba = torch.exp(prediction)
    class_label = CLASSES[proba.argmax()]
    return {"class": int(proba.argmax()), "class_label": class_label, "probability": proba.max().item()}


if __name__ == '__main__':

    # Prepare dataset
    train_dl, val_dl, test_dl = prepare_dataset(batch_size=256, img_height=64, img_weight=64)

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init model
    model = SimpleNet(input_size=4096).to(device)
    # model = AdvancedNet(input_size=4096).to(device)
    # model = RhombusNet(input_size=4096).to(device)

    # Load checkpoints
    model.load_state_dict(torch.load('./models/cloth_model_simple.pth'))

    # Get random image from dataset
    image_index = random.randrange(0, len(test_dl))
    img_orig, img_vec, label = prepare_image(test_dl, index=image_index)

    # Male prediction
    prediction = inference(model, img_vec)
    explain(img_orig, label, prediction)
    print(prediction)

    # Prepare result
    result = format(prediction)
    print(result)
