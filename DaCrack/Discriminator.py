import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DomainDiscriminator, self).__init__()

        # def __init__(self, encoder=vgg, decoder=decoder, decoder2=decoder2):
        #     super(Net, self).__init__()
        # Multi-scale convolutional layers with different dilation rates
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(input_channels, 64, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(input_channels, 64, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(input_channels, 64, kernel_size=3, dilation=4, padding=4)

        # 1x1 convolution to combine the features
        self.conv1x1 = nn.Conv2d(64 * 4, 128, kernel_size=1)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply the multi-scale convolutions
        # print(x.shape)
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))
        out4 = F.relu(self.conv4(x))

        # Concatenate the features along the channel dimension
        out = torch.cat([out1, out2, out3, out4], dim=1)

        # Apply the 1x1 convolution
        out = F.relu(self.conv1x1(out))

        # Global average pooling
        out = self.global_pool(out)

        # Flatten the output for the fully connected layer
        out = torch.flatten(out, 1)

        # Fully connected layer for classification
        out = self.fc(out)

        return out


# # Example usage
# input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor (batch_size, channels, height, width)
# model = DomainDiscriminator(input_channels=3, num_classes=2)  # Assuming binary classification
# output = model(input_tensor)
#
# print(output.shape)
