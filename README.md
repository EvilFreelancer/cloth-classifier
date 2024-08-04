# Simple Clothing Classifier 

Pure TorchVision-based implementation of clothing classification.

## Features

* 64px per 64px input images supported
* Train a model on Fashion MNIST dataset
* Multithreading support (may handle multiple requests at the same time)
* 10 basic classes (limited due to Fashion MNIST)

## Requirements

Before you begin, ensure you have a machine with an GPU that supports modern CUDA, due to the computational
demands of the docker image.

* Nvidia GPU
* CUDA
* Docker
* Docker Compose
* Nvidia Docker Runtime

For detailed instructions on how to prepare a Linux machine for running neural networks, including the installation of
CUDA, Docker, and Nvidia Docker Runtime, please refer to the
publication "[How to Prepare Linux for Running and Training Neural Networks? (+ Docker)](https://dzen.ru/a/ZVt9kRBCTCGlQqyP)"
on Russian.

## Installation

1. Clone the repo and switch to sources root:

   ```shell
   git clone https://github.com/EvilFreelancer/cloth-classifier.git
   cd cloth-classifier
   ```

2. Copy the provided Docker Compose template:

    ```shell
    cp docker-compose.dist.yml docker-compose.yml
    ```

3. Build the Docker image:

    ```shell
    docker-compose build
    ```

4. Start the services:

    ```shell
    docker-compose up -d
    ```

## How to use

```shell
curl \
  "http://localhost:8080/predict" \
  -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tshirt.jpg"
```

```json
{
  "class": 0,
  "class_label": "T-shirt",
  "probability": 2.12345
}
```

## Available classes

```python
CLASSES = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]
```

## Train model

Login to container:

```shell
docker-compose exec app bash
```

Run training script:

```shell
python3 /app/train_model.py
```

It will train a model and put it to `./models/cloth_model_simple.pth`.

## Links

* https://github.com/sssingh/fashion-mnist-classification/blob/master/fashion_mnist_classification_nn_pytorch.ipynb
* https://github.com/roboflow/notebooks/blob/main/notebooks/train-vision-transformer-classification-on-custom-data.ipynb
