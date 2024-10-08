import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassificationCNN(nn.Module):
    def __init__(self):
        super(BinaryClassificationCNN, self).__init__()

        # First convolutional layer that accepts a single grayscale image channel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: (32, 224, 224)
        self.pool1 = nn.MaxPool2d(2, 2)                          # Output: (32, 112, 112)
        self.dropout1 = nn.Dropout(0.25)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Output: (64, 112, 112)
        self.pool2 = nn.MaxPool2d(2, 2)                          # Output: (64, 56, 56)
        self.dropout2 = nn.Dropout(0.25)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Output: (128, 56, 56)
        self.pool3 = nn.MaxPool2d(2, 2)                           # Output: (128, 28, 28)
        self.dropout3 = nn.Dropout(0.25)

        # Dynamically calculate input features for the first fully connected layer
        # For an input of 224x224, after three pooling layers it should be 128x28x28
        self.fc1_in_features = 128 * 28 * 28
        self.fc1 = nn.Linear(self.fc1_in_features, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)  # Final output layer

    def forward(self, x):
        # Apply convolutions, max pooling and dropout
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten the layer to fit into the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.13.0', 'resnet18', weights='IMAGENET1K_V1')
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.state_dict()['conv1.weight'] = self.model.state_dict()['conv1.weight'].sum(dim=1, keepdim=True)
        self.model.fc = nn.Sequential(
            torch.nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x