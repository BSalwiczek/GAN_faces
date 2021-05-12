import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
# import matplotlib.pyplot as plt
from models.GAN import GAN


class DCGAN(GAN):
    def __init__(self, input_shape, z_dim, num_epochs, data_loader, lr, architecture):
        super().__init__(input_shape, z_dim, num_epochs, data_loader, lr, architecture)

    # Train discriminator: maximize log(D(x)) + log(1 - D(G(z)))
    def train_discriminator(self, data, criterion, label, fake_label):
        # set gradient to zero
        self.discriminator.zero_grad()

        batch_size = data.size()[0]

        output = self.discriminator(data).view(-1)
        loss_discriminator_real = criterion(output, label)
        loss_discriminator_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
        generated_data = self.generator(noise)  # Generate fake images with generator
        label.fill_(fake_label)
        output = self.discriminator(generated_data.detach()).view(-1)  # Classify all generated image with discriminator
        loss_discriminator_fake = criterion(output, label)
        loss_discriminator_fake.backward()
        D_G_z1 = output.mean().item()

        # Compute error of discriminator as sum over the fake and the real batches
        loss_discriminator = loss_discriminator_real + loss_discriminator_fake

        # Update discriminator
        self.optimizer_discriminator.step()

        return loss_discriminator, D_x, D_G_z1

    # Train generator: maximize log(D(G(z)))
    def train_generator(self, data, criterion, label, real_label):
        batch_size = data.size()[0]
        noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
        generated_data = self.generator(noise)

        self.generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = self.discriminator(generated_data).view(-1)
        loss_generator = criterion(output, label)  # Calculate generator loss
        loss_generator.backward()

        D_G_z2 = output.mean().item()

        # Update generator
        self.optimizer_generator.step()

        return D_G_z2, loss_generator

    def train(self, save_dir, starting_epoch, starting_iter):
        self.try_load_weights(starting_epoch, save_dir, starting_iter)

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # noise that will be used to see generator progression
        fixed_noise = torch.randn(128, self.z_dim, 1, 1, device=self.device)

        # Convention for real and fake labels
        real_label = 1.
        fake_label = 0.

        iters = starting_iter

        if starting_epoch > 0:
            self.G_losses = np.load(save_dir + "generator_losses.npy").tolist()
            self.D_losses = np.load(save_dir + "discriminator_losses.npy").tolist()

        print("Start training...")

        for epoch in range(starting_epoch, self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.data_loader, 0):
                # send data to device (CUDA or CPU)
                data = data[0].to(self.device)
                batch_size = data.size()[0]

                label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)

                loss_discriminator, D_x, D_G_z1 = self.train_discriminator(data, criterion, label, fake_label)

                D_G_z2, loss_generator = self.train_generator(data, criterion, label, real_label)

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.num_epochs, i, len(self.data_loader),
                             loss_discriminator.item(), loss_generator.item(), D_x, D_G_z1, D_G_z2))

                self.G_losses.append(loss_generator.item())
                self.D_losses.append(loss_discriminator.item())

                self.save_training_progress(epoch, fixed_noise, iters, save_dir, i)

                iters += 1
