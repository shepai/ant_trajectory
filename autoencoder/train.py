import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from model import FlexibleAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing
transform = transforms.ToTensor()

# Load MNIST dataset
images = np.load("/its/home/drs25/ant_trajectory/autoencoder/allangles.npy")  # shape: (N, H, W) or (N, 1, H, W)
images=images.reshape(len(images),1,*images.shape[1:]).astype(np.float32) / 255.0
train_dataset=torch.tensor(images)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model, Loss, Optimizer
model = FlexibleAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Visualize Results
def imshow(imgs, title=""):
    grid = vutils.make_grid(imgs, nrow=8, normalize=True)
    npimg = grid.permute(1, 2, 0).cpu().detach().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(npimg, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

torch.save(model.state_dict(), "/its/home/drs25/ant_trajectory/autoencoder/flexiencoder_weights.pth")

# Show a few original and reconstructed images
model.eval()
with torch.no_grad():
    test_imgs = next(iter(train_loader))
    test_imgs = test_imgs.to(device)
    recon = model(test_imgs)

    imshow(test_imgs[:16], "Original Images")
    imshow(recon[:16], "Reconstructed Images")


