import torch
import torch.nn.functional as F
import torch.nn as nn
from prettytable import PrettyTable


class TimbreCNN(nn.Module):
    def __init__(self):
        super(TimbreCNN, self).__init__()
        self.cnn_layers_v1 = nn.Sequential(
            # Loosely based on fully convolutional variation of LeNet
            # TODO: add BatchNorm layers after every conv layer
            # Input size: (3, 172, 172)
            # conv1
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv2
            nn.Conv2d(in_channels=18, out_channels=45, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv3
            nn.Conv2d(in_channels=45, out_channels=100, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv4
            nn.Conv2d(in_channels=100, out_channels=45, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv5
            nn.Conv2d(in_channels=45, out_channels=16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # Extra layer to further reduce map size to 1
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2),
            # Convert this output to a decision between 0 and 1 with sigmoid
            # activation for binary classification output
            nn.Sigmoid()
            )

        self.cnn_layers_v2 = nn.Sequential(
            # Loosely based on fully convolutional variation of LeNet
            # TODO: add BatchNorm layers after every conv layer
            # Input size: (3, 80, 222)
            # conv1: 3x80x222 to 18x39x109
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=(3, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv2: 18x39x109 to 45x18x52
            nn.Conv2d(in_channels=18, out_channels=45, kernel_size=(3, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv3: 45x18x52 to 100x8x24
            nn.Conv2d(in_channels=45, out_channels=100, kernel_size=(3, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv4: 100x8x24 to 45x3x10
            nn.Conv2d(in_channels=100, out_channels=45, kernel_size=(3, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv5: 45x3x10 to 16x1x3
            nn.Conv2d(in_channels=45, out_channels=16, kernel_size=(3, 5)),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReLU(),
            # Extra layer to further reduce map size to 1
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 3)),
            # Convert this output to a decision between 0 and 1 with sigmoid
            # activation for binary classification output
            nn.Sigmoid()
        )

        self.cnn_layers = nn.Sequential(
            # Loosely based on fully convolutional variation of LeNet
            # TODO: add BatchNorm layers after every conv layer
            # Input size: (1, 300, 221)
            # conv1: 1x300x221 to 6x148x109
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv2: 6x148x109 to 18x72x52
            nn.Conv2d(in_channels=6, out_channels=18, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv3: 18x72x52 to 45x34x24
            nn.Conv2d(in_channels=18, out_channels=45, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv4: 45x34x24 to 18x15x10
            nn.Conv2d(in_channels=45, out_channels=18, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv5: 18x15x10 to 6x5x3
            nn.Conv2d(in_channels=18, out_channels=6, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # Extra layer to further reduce map size to 1
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(5, 3)),
            # Convert this output to a decision between 0 and 1 with sigmoid
            # activation for binary classification output
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 3:  # If single-channel spectrogram, insert the "channel" dimension for input to CNN
            x = torch.unsqueeze(x, 1)
        out = self.cnn_layers(x)
        return out.view(-1)

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params