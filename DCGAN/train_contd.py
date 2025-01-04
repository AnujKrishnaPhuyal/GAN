import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import optim
from DCGAN.model import Generator,Discriminator
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


step = 0
epochs = 210
# Load the checkpoint to resume training
chk="DCGAN\checkpoint_epoch_80.pth"
checkpoint = torch.load(chk)  # Load the latest checkpoint

path = "DCGAN/DCGAN.py"
BATCH_SIZE = 64
LR = 0.0002
image_size = 64
img_channels = 3
Discriminator_features = 64
Generator_features = 64
Noise_dimensions = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data ="BasicGAN/data/img_align_celeba.zip"
# # Load the state_dicts from the checkpoint
generator=Generator(Noise_dimensions,3,3,Generator_features)
discriminator=Discriminator(3,Discriminator_features)

criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

fixed_noise = torch.randn(32, Noise_dimensions, 1, 1).to(device)
writer_real = SummaryWriter(f"DCGAN/logs/real")
writer_fake = SummaryWriter(f"DCGAN/logs/fake")
step = 0



generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

# Optionally, load other information like loss and epoch
loss_D = checkpoint['loss_D']
loss_G = checkpoint['loss_G']
start_epoch = checkpoint['epoch']  # Start from the next epoch
print(f"Resuming training from epoch {start_epoch}")  # Print the start_epoch)
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

# Set models to training mode
generator.train()
discriminator.train()

# Now, continue the training loop from the saved epoch
for epoch in range(start_epoch, epochs):
    print(f"Epoch {epoch}/{epochs}")
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
            'epoch': epoch +1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
            'loss_D': loss_Discriminator,
            'loss_G': loss_Generator,
             'step': step
        }, f"checkpoint_epoch_{epoch}.pth")      