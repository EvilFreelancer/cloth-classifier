import torch
import torch.nn as nn


class RhombusNet(nn.Module):
    """
    Ромбовидная модель с двумя ветвями, dropout и логарифмической функцией активации softmax.

    28px X 28px = 784
    64px X 64px = 4096
    """
    def __init__(self, input_size: int = 784, output_size: int = 10, dropout_rate: float = 0.25):
        super(RhombusNet, self).__init__()

        # Входной слой
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        # Первая ветвь
        self.fc2_1 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2_1 = nn.ReLU()
        self.drop2_1 = nn.Dropout(dropout_rate)
        self.fc3_1 = nn.Linear(input_size // 4, input_size // 8)
        self.relu3_1 = nn.ReLU()
        self.drop3_1 = nn.Dropout(dropout_rate)
        self.fc4_1 = nn.Linear(input_size // 8, input_size // 16)
        self.relu4_1 = nn.ReLU()
        self.drop4_1 = nn.Dropout(dropout_rate)

        # Вторая ветвь
        self.fc2_2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2_2 = nn.ReLU()
        self.drop2_2 = nn.Dropout(dropout_rate)
        self.fc3_2 = nn.Linear(input_size // 4, input_size // 8)
        self.relu3_2 = nn.ReLU()
        self.drop3_2 = nn.Dropout(dropout_rate)
        self.fc4_2 = nn.Linear(input_size // 8, input_size // 16)
        self.relu4_2 = nn.ReLU()
        self.drop4_2 = nn.Dropout(dropout_rate)

        # Объединение ветвей
        self.fc5 = nn.Linear(input_size // 8, input_size // 16)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(dropout_rate)

        # Выходной слой
        self.output = nn.Linear(input_size // 16, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Входной слой
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)

        # Первая ветвь
        x1 = self.relu2_1(self.fc2_1(x))
        x1 = self.drop2_1(x1)
        x1 = self.relu3_1(self.fc3_1(x1))
        x1 = self.drop3_1(x1)
        x1 = self.relu4_1(self.fc4_1(x1))
        x1 = self.drop4_1(x1)

        # Вторая ветвь
        x2 = self.relu2_2(self.fc2_2(x))
        x2 = self.drop2_2(x2)
        x2 = self.relu3_2(self.fc3_2(x2))
        x2 = self.drop3_2(x2)
        x2 = self.relu4_2(self.fc4_2(x2))
        x2 = self.drop4_2(x2)

        # Объединение ветвей
        x = torch.cat((x1, x2), dim=1)
        x = self.relu5(self.fc5(x))
        x = self.drop5(x)

        # Выходной слой
        x = self.output(x)
        x = self.logsoftmax(x)
        return x
