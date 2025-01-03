import torch
import matplotlib.pyplot as plt

class AdversarialNetwork(torch.nn.Module):
    input_height = None
    input_width = None
    input_channels = None
    kernel_size = None
    stride = 1
    padding = 0
    def __init__(self, input_height, input_width, input_channels, kernel_size) -> None:
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
    
    def get_activation_function(self, activation):
        if activation == 'relu':
            return torch.nn.ReLU()
        elif activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            return torch.nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            return torch.nn.Tanh()   
        else:
            raise ValueError('Invalid activation function')
    
    def produce_convolution_layer(self, input_channels, output_channels, kernel_size, stride, padding):
        return torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)

    def produce_pooling_layer(self, kernel_size, stride, padding):
        return torch.nn.MaxPool2d(kernel_size, stride, padding)
    
    def produce_fully_connected_layer(self, input_size, output_size):
        return torch.nn.Linear(input_size, output_size)
    
    def produce_loss_function(self, loss_function):
        if loss_function == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_function == 'bce':
            return torch.nn.BCELoss()
        else:    
            raise ValueError('Invalid loss function')
        
    def produce_optimizer(self, optimizer, model, learning_rate):
        if optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        elif optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        else:   
            raise ValueError('Invalid optimizer')

    def compute_flatten_size(self, conv_output_height, conv_output_width, num_channels):
        return conv_output_height * conv_output_width * num_channels
    

    def create_discriminator(self, d, activation='leaky_relu'):
        """
        Discriminator for MNIST-sized (28×28) grayscale images.
        Downsamples 28×28 -> 1×1 in four steps, output shape [N,1].
        """
        activation_function = self.get_activation_function(activation)

        model = torch.nn.Sequential(
            # 1) 28 -> 14
            torch.nn.Conv2d(self.input_channels, d, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # 2) 14 -> 7
            torch.nn.Conv2d(d, d * 2, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(d * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # 3) 7 -> 4 (note kernel=3, stride=2, pad=1)
            torch.nn.Conv2d(d * 2, d * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(d * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # 4) 4 -> 1
            torch.nn.Conv2d(d * 4, 1, kernel_size=4, stride=1, padding=0),
            torch.nn.Flatten(),  # -> [N, 1]
            torch.nn.Sigmoid()
        )
        return model
    
    def create_generator(self, latent_dim=100, hidden=64, activation='leaky_relu'):
        """
        Generator that outputs 28×28 grayscale images.
        Starts from a latent noise [N, latent_dim, 1, 1].
        """
        activation_function = self.get_activation_function(activation)

        model = torch.nn.Sequential(
            # 1) 1×1 -> 4×4
            torch.nn.ConvTranspose2d(latent_dim, hidden * 4, kernel_size=4, stride=1, padding=0),
            torch.nn.BatchNorm2d(hidden * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # 2) 4×4 -> 7×7 (use kernel=3, stride=2, pad=1)
            torch.nn.ConvTranspose2d(hidden * 4, hidden * 2, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # 3) 7×7 -> 14×14
            torch.nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # 4) 14×14 -> 28×28
            torch.nn.ConvTranspose2d(hidden, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()  # final activation in [-1, 1]
        )
        return model
    
    def debug_discriminator_output(self, discriminator, images):
        x = images
        with torch.no_grad():
            for i, layer in enumerate(discriminator):
                x = layer(x)
                print(f"After layer {i} ({layer}): {x.shape}")

    def train_gan(self, generator, discriminator, dataloader, num_epochs, latent_dim, learning_rate=0.0002):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = generator.to(device)
        discriminator = discriminator.to(device)

        criterion = self.produce_loss_function('bce')
        optimizer_g = self.produce_optimizer('adam', generator, 0.0005)
        optimizer_d = self.produce_optimizer('adam', discriminator, 0.0001)

        for epoch in range(num_epochs):
            for real_images, _ in dataloader:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_images = generator(noise)

                #print("\n[DEBUG] real_images:", real_images.shape)       # Expect [64, 1, 28, 28]
                #print("[DEBUG] fake_images:", fake_images.shape)         # Expect [64, 1, 28, 28]

                real_output = discriminator(real_images)
                #print("[DEBUG] D(real_images):", real_output.shape)      # Expect [64, 1]
                fake_output = discriminator(fake_images.detach())
                #print("[DEBUG] D(fake_images):", fake_output.shape)      # Expect [64, 1]

                real_labels = torch.full((batch_size, 1), 0.95, device=device)
                fake_labels = torch.full((batch_size, 1), 0.05, device=device)
                #print("[DEBUG] real_labels:", real_labels.shape)         # [64, 1]
                #print("[DEBUG] fake_labels:", fake_labels.shape)         # [64, 1]

                real_loss = criterion(real_output, real_labels)
                fake_loss = criterion(fake_output, fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                g_loss = criterion(discriminator(fake_images), real_labels)
                g_loss.backward()
                optimizer_g.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")   
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                    fake_images = generator(noise).cpu()
                    
                    # Plot generated images
                    fig, axes = plt.subplots(1, 8, figsize=(24, 4))
                    for i, ax in enumerate(axes):
                        ax.imshow(fake_images[i][0], cmap='gray')
                        fig.savefig(f'gan_images/{epoch+1}.svg')
                        ax.axis('off')
                    plt.show()

