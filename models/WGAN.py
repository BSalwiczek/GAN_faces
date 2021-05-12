from torch import nn
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np
from models.utils import gradient_penalty

from models.GAN import GAN


class WGAN(GAN):
    def __init__(self, input_shape, z_dim, num_epochs, data_loader, lr, architecture):
        super().__init__(input_shape, z_dim, num_epochs, data_loader, lr, architecture)

        self.CRITIC_ITERS = 5
        self.GP_WEIGHT = 10

        self.GP_losses = []

    def train(self, save_dir, starting_epoch, starting_iter):
        self.try_load_weights(starting_epoch, save_dir, starting_iter)

        # noise that will be used to see generator progression
        fixed_noise = torch.randn(128, self.z_dim, 1, 1, device=self.device)

        if starting_epoch > 0:
            self.G_losses = np.load(save_dir + "generator_losses.npy").tolist()
            self.D_losses = np.load(save_dir + "discriminator_losses.npy").tolist()

        iters = starting_iter

        print("Start training...")

        for epoch in range(starting_epoch, self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.data_loader, 0):
                # send data to device (CUDA or CPU)
                data = data[0].to(self.device)

                self.train_discriminator(data)

                if iters % self.CRITIC_ITERS == 0:
                    self.train_generator(data)

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_GP(x): %.4f'
                          % (epoch, self.num_epochs, i, len(self.data_loader),
                             self.D_losses[-1], self.G_losses[-1], self.GP_losses[-1]))

                self.save_training_progress(epoch, fixed_noise, iters, save_dir, i)

                iters += 1

    def train_discriminator(self, data):
        batch_size = data.size()[0]
        noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)

        generated_data = self.generator(noise)  # Generate fake image batch with G
        # data = Variable(data).cuda()

        discriminator_real = self.discriminator(data).reshape(-1)
        discriminator_fake = self.discriminator(generated_data).reshape(-1)

        gradient_p = gradient_penalty(self.discriminator, data, generated_data) * self.GP_WEIGHT
        self.GP_losses.append(gradient_p.item())

        discriminator_loss = torch.mean(discriminator_fake) - torch.mean(discriminator_real) + gradient_p
        self.D_losses.append(discriminator_loss.item())

        self.optimizer_discriminator.zero_grad()
        discriminator_loss.backward()
        self.optimizer_discriminator.step()

    def train_generator(self, data):
        batch_size = data.size()[0]

        noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
        generated_data = self.generator(noise)

        gen_fake = self.discriminator(generated_data).reshape(-1)

        generator_loss = -gen_fake.mean()
        self.G_losses.append(generator_loss.item())

        self.optimizer_generator.zero_grad()
        generator_loss.backward()
        self.optimizer_generator.step()
