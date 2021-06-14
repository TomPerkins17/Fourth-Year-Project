import torch
import torch.nn.functional as F
import torch.nn as nn
from prettytable import PrettyTable


class SingleNoteTimbreCNN(nn.Module):
    def __init__(self):
        super(SingleNoteTimbreCNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Loosely based on fully convolutional variation of LeNet
            # Input size: (1, 300, 222)
            # conv1: 1x300x221 to 6x148x109
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv2: 6x148x109 to 18x72x52
            nn.Conv2d(in_channels=6, out_channels=18, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(18),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv3: 18x72x52 to 45x34x24
            nn.Conv2d(in_channels=18, out_channels=45, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(45),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv4: 45x34x24 to 18x15x10
            nn.Conv2d(in_channels=45, out_channels=18, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(18),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv5: 18x15x10 to 6x5x3
            nn.Conv2d(in_channels=18, out_channels=6, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # Extra layer to further reduce map size to 1
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(5, 3), bias=False),
            nn.BatchNorm2d(1),
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


class SingleNoteTimbreCNNSmall(SingleNoteTimbreCNN):
    def __init__(self):
        super(SingleNoteTimbreCNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Reduces number and size of convolutional units by using strided convolutions and larger
            # Increased number of filters per conv layer (deeper and narrower architecture)
            # Cool idea in http://www.ofai.at/~jan.schlueter/pubs/2016_ismir.pdf is applying wider pooling to the
            # first axis (e.g. 4x1) to render the network more pitch invariant.
            # Input size: (1, 300, 222)
            # conv1: 1x300x222 to 16x149x110
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv2: 16x149x109 to 32x49x54
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(3, 2)),
            nn.ReLU(),
            # conv3: 32x49x53 to 32x15x26
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(3, 2)),
            nn.ReLU(),
            # conv4: 32x15x25 to 16x4x8
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            # Extra layer to further reduce map size to 1
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(4, 5), bias=False),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(kernel_size=(1, 4)),
            # Convert this output to a decision between 0 and 1 with sigmoid
            # activation for binary classification output
            nn.Sigmoid()
        )


class MelodyTimbreCNN(SingleNoteTimbreCNN):
    def __init__(self):
        super(SingleNoteTimbreCNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Loosely based on fully convolutional variation of LeNet
            # TODO: Convert to 3-channel input for context
            # Input size: (1, 300, 222)
            # conv1: 1x300x221 to 6x148x109
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv2: 6x148x109 to 18x72x52
            nn.Conv2d(in_channels=6, out_channels=18, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(18),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv3: 18x72x52 to 45x34x24
            nn.Conv2d(in_channels=18, out_channels=45, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(45),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv4: 45x34x24 to 18x15x10
            nn.Conv2d(in_channels=45, out_channels=18, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(18),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # conv5: 18x15x10 to 6x5x3
            nn.Conv2d(in_channels=18, out_channels=6, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # Extra layer to further reduce map size to 1
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(5, 3), bias=False),
            nn.BatchNorm2d(1),
            # Convert this output to a decision between 0 and 1 with sigmoid
            # activation for binary classification output
            nn.Sigmoid()
        )
