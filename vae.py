import torch
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        size = int((input.size(1) // self.n_channels)**0.5)
        return input.view(input.size(0), self.n_channels, size, size)


class BaseVAE(nn.Module):
    """
    Base abstract class for the Variational Autoencoders
    """
    def __init__(self, channels=1, width=28, height=28, z_dim=2):
        """
        Constructor

        Parameters:
            channels - The number of channels for the image
            width - The width of the image in pixels
            height - The height of the image in pixels
            z_dim - The dimension of the latent space
        """
        super(BaseVAE, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.z_dim = z_dim

    def getNbChannels(self):
        """
        Returns the number of channels of the handled images
        """
        return self.channels

    def getWidth(self):
        """
        Returns the width of the handled images in pixels
        """
        return self.width

    def getHeight(self):
        """
        Returns the height of the handled images in pixels
        """
        return self.height

    def getZDim(self):
        """
        Returns the dimension of the latent space of the VAE
        """
        return self.z_dim

    def flatten(self, x):
        """
        Can be used to flatten the output image. This method will only handle
        images of the original size specified for the network
        """
        return x.view(-1, self.channels * self.width * self.height)

    def unflatten(self, x):
        """
        Can be used to unflatten an image handled by the network. This method
        will only handle images of the original size specified for the network
        """
        return x.view(-1, self.channels, self.width, self.height)


class FCVAE(BaseVAE):
    """
    Fully connected Variational Autoencoder
    """
    def __init__(self, channels=1, width=28, height=28, z_dim=2):
        super(FCVAE, self).__init__(channels, width, height, z_dim)

        self.fc1 = nn.Linear(self.channels * self.width * self.height, 400)
        self.fc21 = nn.Linear(400, self.z_dim)
        self.fc22 = nn.Linear(400, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 400)
        self.fc4 = nn.Linear(400, self.channels * self.width * self.height)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(self.flatten(x))
        z = self.reparameterize(mu, logvar)
        return self.unflatten(self.decode(z)), mu, logvar


class ConvVAE(BaseVAE):
    """
    Convolutional Variational Autoencoder
    """
    def __init__(self, channels=3, width=28, height=28, z_dim=32):
        super(ConvVAE, self).__init__(channels, width, height, z_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            Flatten())

        dummy_input = torch.ones([1, self.channels, self.width, self.height])
        h_dim = self.encoder(dummy_input).size(1)

        self.fc1 = nn.Linear(h_dim, self.z_dim)
        self.fc2 = nn.Linear(h_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, self.channels, kernel_size=4, stride=1),
            nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar
