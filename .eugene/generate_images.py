import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Define a more complex Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# Configuration parameters
latent_dim = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
num_samples = 10
random_seed = 42

# Set the random seed for reproducibility
torch.manual_seed(random_seed)

# Initialize the generator model
generator = Generator(latent_dim).to(device)


# Initialize KSampler with the generator model
class KSampler:
    def __init__(self, model, latent_dim, device="cuda"):
        self.model = model
        self.latent_dim = latent_dim
        self.device = device
        self.model.to(self.device)

    def sample_latent(self, num_samples):
        return torch.randn(num_samples, self.latent_dim).to(self.device)

    def generate_images(self, latent_vectors):
        with torch.no_grad():
            images = self.model(latent_vectors)
        return images


sampler = KSampler(generator, latent_dim, device)

# Configure sampling parameters
latent_vectors = sampler.sample_latent(num_samples)

# Generate images
generated_images = sampler.generate_images(latent_vectors)

# Visualize generated images
fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
for i, img in enumerate(generated_images):
    img = img.squeeze().cpu().numpy()
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
plt.show()
