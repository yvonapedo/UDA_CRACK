import torch
import torch.nn as nn


class ForegroundEnhancementModule(nn.Module):
    def __init__(self, encoder):
        super(ForegroundEnhancementModule, self).__init__()
        self.encoder = encoder

    def forward(self, x_s, x_t):
        """
        Forward pass for the Foreground Enhancement Module (FEM).

        Parameters:
            x_s (torch.Tensor): Source domain image.
            x_t (torch.Tensor): Target domain image.

        Returns:
            P_bt_F_ct (torch.Tensor): Target background with source cracks.
            P_bs_F_cs (torch.Tensor): Source background with target cracks.
        """
        # Get shallow and deep features from the encoder
        F_bs, F_cs = self.extract_features(x_s)
        F_bt, F_ct = self.extract_features(x_t)

        # Perform background exchange
        P_bt_F_cs = torch.cat([F_bt, F_cs], dim=1)  # Target background with source cracks
        P_bs_F_ct = torch.cat([F_bs, F_ct], dim=1)  # Source background with target cracks

        return P_bt_F_cs, P_bs_F_ct

    def extract_features(self, x):
        """
        Extract shallow and deep features from the encoder.

        Parameters:
            x (torch.Tensor): Input image.

        Returns:
            F_b (torch.Tensor): Low-level background features (shallow features).
            F_c (torch.Tensor): High-level crack features (deep features).
        """
        # Assuming the encoder has shallow (low-level) and deep (high-level) layers
        shallow_features = []
        deep_features = []

        for layer in self.encoder.children():
            x = layer(x)
            if len(shallow_features) < 2:  # Assuming first 2 layers are shallow
                shallow_features.append(x)
            else:
                deep_features.append(x)

        F_b = shallow_features[-1]  # Last shallow feature map
        F_c = deep_features[-1]  # Last deep feature map

        return F_b, F_c


# Example encoder (this is just a simple placeholder; use your actual model encoder)
class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Example usage
# encoder = SimpleEncoder()
# fem = ForegroundEnhancementModule(encoder)
#
# # Example inputs (source and target domain images)
# x_s = torch.randn(1, 3, 256, 256)  # Source domain image
# x_t = torch.randn(1, 3, 256, 256)  # Target domain image
#
# # Run the FEM
# P_bt_F_cs, P_bs_F_ct = fem(x_s, x_t)
#
# print("Target background with source cracks shape:", P_bt_F_cs.shape)
# print("Source background with target cracks shape:", P_bs_F_ct.shape)
