import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
# import matplotlib.pyplot as plt


# initialize weights mean = 0, stdev = 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN:
    def __init__(self, input_shape, z_dim, num_epochs, data_loader, lr, architecture):
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.lr = lr
        self.beta1 = 0.5
        self.z_dim = z_dim

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # creating discriminator
        self.discriminator = Discriminator(architecture, input_shape).to(self.device)
        self.discriminator.apply(weights_init)

        # creating generator
        self.generator = Generator(architecture, input_shape, z_dim).to(self.device)
        self.generator.apply(weights_init)

    def train_discriminator_real(self, data, real_label, criterion, real_cpu, label):
        output = self.discriminator(real_cpu).view(-1)  # Forward pass real batch through D
        error_discriminator_real = criterion(output, label)  # Calculate loss on all-real batch
        error_discriminator_real.backward()  # Calculate gradients for D in backward pass
        D_x = output.mean().item()

        return D_x, error_discriminator_real

    def train_discriminator_fake(self, b_size, label, fake_label, criterion):
        noise = torch.randn(b_size, self.z_dim, 1, 1, device=self.device)
        fake = self.generator(noise)  # Generate fake image batch with G
        label.fill_(fake_label)
        output = self.discriminator(fake.detach()).view(-1)  # Classify all fake batch with D
        error_discriminator_fake = criterion(output, label)  # Calculate D's loss on the all-fake batch
        error_discriminator_fake.backward()  # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        D_G_z1 = output.mean().item()

        return D_G_z1, error_discriminator_fake, fake

    def train_generator(self, label, real_label, fake, criterion):
        self.generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = self.discriminator(fake).view(
            -1)  # Since we just updated D, perform another forward pass of all-fake batch through D
        error_generator = criterion(output, label)  # Calculate G's loss based on this output
        error_generator.backward()  # Calculate gradients for G

        D_G_z2 = output.mean().item()

        return D_G_z2, error_generator

    def train(self, save_dir, starting_epoch):
        if starting_epoch > 0:
            #load latest weights
            self.discriminator.load_state_dict(torch.load(save_dir+"discriminator_epoch_1_iter_6000"))
            self.discriminator.eval()

            self.generator.load_state_dict(torch.load(save_dir + "generator_epoch_1_iter_6000"))
            self.generator.eval()


        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        fixed_noise = torch.randn(128, self.z_dim, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizer_generator = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        if starting_epoch > 0:
            G_losses = np.load(save_dir+"generator_losses.npy").tolist()
            D_losses = np.load(save_dir + "discriminator_losses.npy").tolist()



        print("Starting Training Loop...")

        for epoch in range(starting_epoch, self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.data_loader, 0):
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)

                # ---------------------------------------------
                # Train discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                # ---------------------------------------------

                # set gradient to zero
                self.discriminator.zero_grad()
                D_x, error_discriminator_real = self.train_discriminator_real(data, real_label, criterion, real_cpu,
                                                                              label)
                D_G_z1, error_discriminator_fake, fake = self.train_discriminator_fake(b_size, label, fake_label, criterion)

                # Compute error of discriminator as sum over the fake and the real batches
                error_discriminator = error_discriminator_real + error_discriminator_fake

                # Update discriminator
                optimizer_discriminator.step()

                # ---------------------------------------------
                # Train generator: maximize log(D(G(z)))
                # ---------------------------------------------
                D_G_z2, error_generator = self.train_generator(label, real_label, fake, criterion)

                # Update generator
                optimizer_generator.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.num_epochs, i, len(self.data_loader),
                             error_discriminator.item(), error_generator.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(error_generator.item())
                D_losses.append(error_discriminator.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs - 1) and (i == len(self.data_loader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                    # save weights
                    torch.save(self.generator.state_dict(), f"{save_dir}generator_epoch_{str(epoch)}_iter_{str(iters)}")
                    torch.save(self.discriminator.state_dict(), f"{save_dir}discriminator_epoch_{str(epoch)}_iter_{str(iters)}")

                if iters % 100 == 0:
                    with open(f"{save_dir}generator_losses.npy",'wb') as f:
                        np.save(f,np.array(G_losses))

                    with open(f"{save_dir}discriminator_losses.npy",'wb') as f:
                        np.save(f,np.array(D_losses))

                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),f"{save_dir}/imgs/epoch_{str(epoch)}_iter_{str(iters)}.png")

                iters += 1

        return G_losses, D_losses, img_list


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class Discriminator(nn.Module):
    def __init__(self,architecture, input_dim):
        super(Discriminator, self).__init__()

        if architecture == 0:
            nc = 3
            ndf = 32
            self.main = nn.Sequential(
                # input is (nc) x 128 x 128
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 64 x 64
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 32 x 32
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 16 x 16
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 8 x 8
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*16) x 4 x 4
                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        if architecture == 1:
            self.main = nn.Sequential(
                nn.Conv2d(3, 48, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(48),

                nn.Conv2d(48, 96, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(96),

                nn.Conv2d(96, 192, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(192),

                nn.Conv2d(192, 384, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(384),

                nn.Conv2d(384, 768, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.BatchNorm2d(768),

                nn.Conv2d(768, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                
            )



    def forward(self, x):
        return self.main(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Generator(nn.Module):
    def __init__(self, architecture, input_dim, z_dim):
        super(Generator, self).__init__()
        nz = z_dim

        if architecture == 0 or architecture == 1:
            ngf = 32
            nc = 3
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf*2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

    def forward(self, x):
        return self.main(x)
