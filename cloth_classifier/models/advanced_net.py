import torch.nn as nn


class AdvancedNet(nn.Module):
    """
    Многослойная полносвязная сеть с dropout и логарифмической функцией активации softmax.
    """

    def __init__(self, input_size: int = 784, output_size: int = 10, dropout_rate: float = 0.25):
        super(AdvancedNet, self).__init__()

        # Входной слой
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        # Скрытые слои
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(input_size // 4, input_size // 8)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(input_size // 8, input_size // 16)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(dropout_rate)

        # Выходной слой
        self.output = nn.Linear(input_size // 16, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Входной слой
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)

        # Скрытые слои
        x = self.relu2(self.fc2(x))
        x = self.drop2(x)
        x = self.relu3(self.fc3(x))
        x = self.drop3(x)
        x = self.relu4(self.fc4(x))
        x = self.drop4(x)

        # Выходной слой
        x = self.output(x)
        x = self.logsoftmax(x)
        return x
