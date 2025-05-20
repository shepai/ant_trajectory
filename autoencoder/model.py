import torch.nn.functional as F
import torch.nn as nn
class FlexibleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # half size
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # quarter size
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # double size
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # original size
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

if __name__ =="__main__":
    import torch
    test_data=torch.rand(10,1,480,200)
    model=FlexibleAutoencoder()
    out=model.forward(test_data)
    print(out.shape)