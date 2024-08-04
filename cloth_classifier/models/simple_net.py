import torch.nn as nn


class SimpleNet(nn.Module):
    """
    Простая полносвязная сеть с dropout и логарифмической функцией активации softmax.
    """

    def __init__(self, input_size: int = 784, output_size: int = 10, dropout_rate: float = 0.25):
        super(SimpleNet, self).__init__()

        # Входной слой
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        # Скрытый слой
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        # Выходной слой
        self.output = nn.Linear(64, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)
        x = self.relu2(self.fc2(x))
        x = self.drop2(x)
        x = self.output(x)
        x = self.logsoftmax(x)
        return x
