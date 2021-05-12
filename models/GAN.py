import torch
from models.Generator import Generator
from models.Discriminator import Discriminator
from models.utils import weights_init
import torchvision.utils as vutils
import numpy as np
import torch.optim as optim

class GAN:
    def __init__(self, input_shape, z_dim, num_epochs, data_loader, lr, architecture):
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.lr = lr
        self.z_dim = z_dim

        self.G_losses = []
        self.D_losses = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # creating discriminator
        self.discriminator = Discriminator(architecture, input_shape).to(self.device)
        self.discriminator.apply(weights_init)

        # creating generator
        self.generator = Generator(architecture, input_shape, z_dim).to(self.device)
        self.generator.apply(weights_init)

        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.99))
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.99))

    def train(self, save_dir, starting_epoch, starting_iter):
        pass

    def save_training_progress(self, epoch, fixed_noise, iters, save_dir,i):
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == self.num_epochs - 1) and (i == len(self.data_loader) - 1)):
            with torch.no_grad():
                fake = self.generator(fixed_noise).detach().cpu()

            # save weights
            torch.save(self.generator.state_dict(), f"{save_dir}generator_epoch_{str(epoch)}_iter_{str(iters)}")
            torch.save(self.discriminator.state_dict(),
                       f"{save_dir}discriminator_epoch_{str(epoch)}_iter_{str(iters)}")

        if iters % 100 == 0:
            with open(f"{save_dir}generator_losses.npy", 'wb') as f:
                np.save(f, np.array(self.G_losses))

            with open(f"{save_dir}discriminator_losses.npy", 'wb') as f:
                np.save(f, np.array(self.D_losses))

            with torch.no_grad():
                fake = self.generator(fixed_noise).detach().cpu()
            vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),
                              f"{save_dir}/imgs/epoch_{str(epoch)}_iter_{str(iters)}.png")

    def try_load_weights(self, starting_epoch, save_dir, starting_iter):
        if starting_epoch > 0:
            # Load latest weights
            self.discriminator.load_state_dict(
                torch.load(save_dir + f"discriminator_epoch_{starting_epoch}_iter_{starting_iter}"))
            self.discriminator.eval()

            self.generator.load_state_dict(
                torch.load(save_dir + f"generator_epoch_{starting_epoch}_iter_{starting_iter}"))
            self.generator.eval()