import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ----------------------------
# Hyperparameters
# ----------------------------
batch_size = 128
lr = 0.0002
z_dim = 100
num_epochs = 50
image_size = 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Data Loader
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# Generator
# ----------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size*image_size),
            nn.Tanh()  # outputs in [-1,1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), 1, image_size, image_size)
        return img

# ----------------------------
# Discriminator
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size*image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# ----------------------------
# Initialize models and optimizers
# ----------------------------
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# ----------------------------
# Training
# ----------------------------
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        # ---------------------
        # Train Discriminator
        # ---------------------
        z = torch.randn(batch_size_curr, z_dim).to(device)
        fake_imgs = generator(z)

        real_labels = torch.ones(batch_size_curr, 1).to(device)
        fake_labels = torch.zeros(batch_size_curr, 1).to(device)

        d_loss_real = criterion(discriminator(real_imgs), real_labels)
        d_loss_fake = criterion(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        g_loss = criterion(discriminator(fake_imgs), real_labels)  # trick discriminator
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # ---------------------
    # Save sample images
    # ---------------------
    if (epoch+1) % 5 == 0:
        z = torch.randn(16, z_dim).to(device)
        sample_imgs = generator(z).detach().cpu()
        sample_imgs = (sample_imgs + 1) / 2  # scale to [0,1]

        fig, axes = plt.subplots(4, 4, figsize=(4,4))
        for ax, img in zip(axes.flatten(), sample_imgs):
            ax.imshow(img.squeeze(), cmap='gray')
            ax.axis('off')
        plt.show()
