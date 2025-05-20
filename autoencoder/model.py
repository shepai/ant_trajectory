import torch.nn.functional as F
import torch.nn as nn
import torch
class FlexibleAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 48, 8), latent_dim=32):
        super().__init__()

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> (16, 4, 24)
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> (32, 2, 12)
            nn.ReLU(True),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy_out = self.encoder_cnn(dummy)
            self._enc_shape = dummy_out.shape[1:]  # (32, 2, 12)
            self.flattened_size = dummy_out.numel()  # 32*2*12 = 768

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(self.flattened_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flattened_size)
        self.unflatten = nn.Unflatten(1, self._enc_shape)

        # Decoder
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> (16, 4, 24)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # -> (1, 8, 48)
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        return self.fc_enc(x)

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.unflatten(x)
        return self.decoder_cnn(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


if __name__ =="__main__":
    import torch
    test_data=torch.rand(10,1,48,8)
    model=FlexibleAutoencoder()
    out=model.forward(test_data)
    print(out.shape)