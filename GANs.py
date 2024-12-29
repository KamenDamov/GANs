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
            return torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:   
            raise ValueError('Invalid optimizer')

    def compute_flatten_size(self, conv_output_height, conv_output_width, num_channels):
        return conv_output_height * conv_output_width * num_channels
    
    def create_discriminator(self, activation='leaky_relu'):
        activation_function = self.get_activation_function(activation)

        # Build discriminator model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_channels, 32, kernel_size=4, stride=2, padding=1),  # Output: 14x14
            activation_function,
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 7x7
            torch.nn.BatchNorm2d(64),
            activation_function,
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 4x4
            torch.nn.BatchNorm2d(128),
            activation_function,
            torch.nn.Flatten(),  # Flatten for linear layer
            torch.nn.LazyLinear(1),  # Automatically infer the input size
            self.get_activation_function('sigmoid'),  # Real/fake probability
        )
        return model

    def create_generator(self, latent_dim, activation='relu'):
        activation_function = self.get_activation_function(activation)

        model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=1, padding=0),  # Output: 4x4
            torch.nn.BatchNorm2d(128),
            activation_function,
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 8x8
            torch.nn.BatchNorm2d(64),
            activation_function,
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            torch.nn.BatchNorm2d(32),
            activation_function,
            torch.nn.ConvTranspose2d(32, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=0),  # Output: 28x28
            self.get_activation_function('tanh'),  # Normalize output to [-1, 1]
        )
        return model

    def train_gan(self, generator, discriminator, dataloader, num_epochs, latent_dim, learning_rate=0.0002):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = generator.to(device)
        discriminator = discriminator.to(device)

        criterion = self.produce_loss_function('bce')
        optimizer_g = self.produce_optimizer('adam', generator, learning_rate)
        optimizer_d = self.produce_optimizer('adam', discriminator, learning_rate)

        for epoch in range(num_epochs):
            for real_images, _ in dataloader:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Generate fake images
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_images = generator(noise)

                # Debug shapes
                #print(f"Real images shape: {real_images.shape}")  # Expected: [batch_size, 1, 28, 28]
                #print(f"Fake images shape: {fake_images.shape}")  # Expected: [batch_size, 1, 28, 28]

                # Train Discriminator
                optimizer_d.zero_grad()
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                real_loss = criterion(discriminator(real_images), real_labels)
                fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                g_loss = criterion(discriminator(fake_images), real_labels)
                g_loss.backward()
                optimizer_g.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


            print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")    
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    noise = torch.randn(16, latent_dim, 4, 4, device=device)
                    fake_images = generator(noise).cpu()
                    fake_images = fake_images.view(-1, 1, 28, 28)
                    
                    # Plot generated images
                    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
                    for i, ax in enumerate(axes):
                        ax.imshow(fake_images[i][0], cmap='gray')
                        ax.axis('off')
                    plt.show()

