"""
AI Creative Technologist GAN Workflow Template
Responsibilities covered:
- Dataset preparation and management
- GAN training and latent space exploration
- Logging and reproducible workflows
- Clear, collaborative structure
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

# -------------------------------
# 1. Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 2. Dataset Preparation
# -------------------------------
# Example: MNIST dataset (replace with custom image/video dataset)
dataset_path = "./data"
transform = transforms.Compose([
    transforms.Resize((64, 64)),       # resize images
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # scale to [-1, 1]
])

dataset = datasets.MNIST(root=dataset_path, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# -------------------------------
# 3. GAN Models
# -------------------------------
latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 64*64),  # output 64x64 image
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), 1, 64, 64)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64*64, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# -------------------------------
# 4. Loss and Optimizers
# -------------------------------
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# -------------------------------
# 5. Training Loop with Logging
# -------------------------------
epochs = 50
log_file = "training_log.txt"

with open(log_file, "w") as log:
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            real_loss = criterion(discriminator(real_imgs), real_labels)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            g_loss = criterion(discriminator(fake_imgs), real_labels)
            g_loss.backward()
            optimizer_G.step()

        # Logging per epoch
        log_line = f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}\n"
        print(log_line, end="")
        log.write(log_line)

        # Save sample images for collaborators to see
        z = torch.randn(16, latent_dim).to(device)
        samples = generator(z).cpu().detach()
        utils.save_image(samples, f"samples_epoch_{epoch+1}.png", nrow=4, normalize=True)

# -------------------------------
# 6. Latent Space Exploration (example)
# -------------------------------
z = torch.randn(1, latent_dim).to(device)
with torch.no_grad():
    generated_img = generator(z).cpu().squeeze()
plt.imshow(generated_img, cmap='gray')
plt.title("Sample Image from Latent Space")
plt.show()

print("Training complete. Logs and sample images saved for collaborators.")
