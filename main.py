import torch
import torch.nn as nn
import torch.optim as optim
from GANs import AdversarialNetwork
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def main():
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Download and load the training dataset
    mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)
    cnn = AdversarialNetwork(input_height=28, input_width=28, input_channels=1, kernel_size=4)

    latent_dim = 100  # Size of the random noise vector

    # Create generator and discriminator
    generator = cnn.create_generator(latent_dim=latent_dim)
    discriminator = cnn.create_discriminator()

    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Train GAN
    cnn.train_gan(generator, discriminator, dataloader, num_epochs=10, latent_dim=latent_dim)
    noise = torch.randn(16, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(noise).cpu()

    # Visualize generated images
    fake_images = fake_images.view(-1, 1, 28, 28)
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(fake_images[i][0], cmap='gray')
        ax.axis('off')
    plt.show()