from model import Generator,Discriminator,initialize_weights
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

BATCH_SIZE = 64
LR = 0.0002
image_size = 64
img_channels = 3
Discriminator_features = 64
Generator_features = 64
Noise_dimensions = 100
epochs = 100


data ="BasicGAN/data/img_align_celeba.zip"

transforms = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)]
        ),
    ]
)

datasets = datasets.ImageFolder(root=data, transform=transforms)

dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)

import torchvision.utils as vutils
# Plot some training images
real_batch = next(iter(dataloader))
print(real_batch[0].shape)
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:32], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()



generator = Generator(Noise_dimensions, img_channels, img_channels, Generator_features).to(device)
discriminator = Discriminator(img_channels, Discriminator_features).to(device)
initialize_weights(generator)
initialize_weights(discriminator)

criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

fixed_noise = torch.randn(32, Noise_dimensions, 1, 1).to(device)
writer_real = SummaryWriter(f"DCGAN/logs/real")
writer_fake = SummaryWriter(f"DCGAN/logs/fake")
step = 0

for epoch in range(epochs):
    for idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Noise_dimensions, 1, 1).to(device)
        # print(real.shape)
        
        fake = generator(noise)
        disc_real = discriminator(real).reshape(-1)
        disc_fake = discriminator(fake).reshape(-1)

        #for discriminator
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_Discriminator = (loss_disc_real + loss_disc_fake) / 2
        discriminator.zero_grad()
        loss_Discriminator.backward(retain_graph=True)
        discriminator_optimizer.step()

        #for generator

        output = discriminator(fake).reshape(-1)
        loss_Generator = criterion(output, torch.ones_like(disc_fake))
        generator.zero_grad()
        loss_Generator.backward()
        generator_optimizer.step()

        if idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {idx}/{len(dataloader)} \
                  Loss D: {loss_Discriminator:.4f}, loss G: {loss_Generator:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:25], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:25], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
    if epoch % 20 == 0: 
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
            'loss_D': loss_Discriminator,
            'loss_G': loss_Generator,
        }, f"checkpoint_epoch_{epoch}.pth")      