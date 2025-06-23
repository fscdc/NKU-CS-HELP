import sys
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")
num_epoch = 10


use_conv = True
is_train = True

# Common codes
def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0, 2).transpose(0, 1)  # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())


def save_imgs(x, filename="output.pdf"):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.numpy())
    plt.axis("off")
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def save_combined_imgs(adjusted_images, filename_prefix="output"):
    for i in range(0, len(adjusted_images), 3):
        grid_images = []
        for j in range(3):
            if i + j < len(adjusted_images):
                grid = vutils.make_grid(
                    adjusted_images[i + j].detach().cpu(),
                    nrow=8,
                    normalize=True,
                    pad_value=0.3,
                )
                grid_images.append(grid)

        combined_grid = torch.cat(grid_images, dim=1).permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 2))
        plt.imshow(combined_grid, cmap="gray")
        plt.axis("off")
        plt.savefig(
            f"{filename_prefix}_{i // 3 + 1}.pdf", format="pdf", bbox_inches="tight"
        )
        plt.close()


## D/G Model
# Original Model
class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), 784)  # flatten (bs x 1 x 28 x 28) -> (bs x 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out)  # range [-1, 1]
        # convert to image
        out = out.view(out.size(0), 1, 28, 28)
        return out


# CNN_Discriminator & CNN_Generator
class CNNDiscriminator(nn.Module):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
        )
        self.activation1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_layer2 = nn.Conv2d(
            64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
        )
        self.batch_norm2 = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.activation2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout_layer = nn.Dropout(p=0.3, inplace=False)
        self.fc_layer = nn.Linear(in_features=6272, out_features=1, bias=True)

    def forward(self, x):
        x = self.activation1(self.conv_layer1(x))
        x = self.activation2(self.batch_norm2(self.conv_layer2(x)))
        x = self.dropout_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return torch.sigmoid(x).view(-1, 1)


class CNNGenerator(nn.Module):
    def __init__(self):
        super(CNNGenerator, self).__init__()
        self.deconv_layer1 = nn.ConvTranspose2d(
            100, 256, kernel_size=(7, 7), stride=(1, 1)
        )
        self.batch_norm1 = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.activation1 = nn.ReLU(inplace=True)
        self.deconv_layer2 = nn.ConvTranspose2d(
            256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
        )
        self.batch_norm2 = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.activation2 = nn.ReLU(inplace=True)
        self.deconv_layer3 = nn.ConvTranspose2d(
            128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.activation1(self.batch_norm1(self.deconv_layer1(x)))
        x = self.activation2(self.batch_norm2(self.deconv_layer2(x)))
        x = torch.tanh(self.deconv_layer3(x))
        return x


if is_train:
    # let's download the Fashion MNIST data, if you do this locally and you downloaded before,
    # you can change data paths to point to your existing files
    # dataset = torchvision.datasets.MNIST(root='./MNISTdata', ...)
    dataset = torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST/",
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
        download=True,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # and the BCE criterion which computes the loss above:
    criterion = nn.BCELoss()

    if not use_conv:
        # instantiate a Generator and Discriminator according to their class definition.
        D = Discriminator()
        print(D)
        G = Generator()
        print(G)

        # ============== No need for these codes in ipynb ================
        # ix = 149
        # x, _ = dataset[ix]
        # plt.matshow(x.squeeze().numpy(), cmap=plt.cm.gray)
        # plt.colorbar()

        # # for one image:
        # Dscore = D(x)
        # Dscore

        # # How you can get a batch of images from the dataloader:
        # xbatch, _ = iter(dataloader).next()  # 64 x 1 x 28 x 28: minibatch of 64 samples
        # xbatch.shape
        # D(xbatch)  # 64x1 tensor: 64 predictions of probability of input being real.
        # D(xbatch).shape

        # show_imgs(xbatch)
        # ========= The code in 3 Intermezzo is not needed here ==========

        # Now let's set up the optimizers
        optimizerD = torch.optim.SGD(D.parameters(), lr=0.01)
        optimizerG = torch.optim.SGD(G.parameters(), lr=0.01)

        # ============== No need for these codes in ipynb ================
        # # STEP 1: Discriminator optimization step
        # x_real, _ = iter(dataloader).next()
        # lab_real = torch.ones(64, 1)
        # lab_fake = torch.zeros(64, 1)
        # # reset accumulated gradients from previous iteration
        # optimizerD.zero_grad()

        # D_x = D(x_real)
        # lossD_real = criterion(D_x, lab_real)

        # z = torch.randn(64, 100) # random noise, 64 samples, z_dim=100
        # x_gen = G(z).detach()
        # D_G_z = D(x_gen)
        # lossD_fake = criterion(D_G_z, lab_fake)

        # lossD = lossD_real + lossD_fake
        # lossD.backward()
        # optimizerD.step()

        # # print(D_x.mean().item(), D_G_z.mean().item())

        # # STEP 2: Generator optimization step
        # # note how only one of the terms involves the Generator so this is the only one that matters for G.
        # # reset accumulated gradients from previous iteration
        # optimizerG.zero_grad()

        # z = torch.randn(64, 100) # random noise, 64 samples, z_dim=100
        # D_G_z = D(G(z))
        # lossG = criterion(D_G_z, lab_real) # -log D(G(z))

        # lossG.backward()
        # optimizerG.step()

        # print(D_G_z.mean().item())
        # ============== No need for these codes in ipynb ================

        # The full training loop
        print("Device: ", device)
        # Re-initialize D, G:
        D = Discriminator().to(device)
        G = Generator().to(device)
        # Now let's set up the optimizers (Adam, better than SGD for this)
        # optimizerD = torch.optim.SGD(D.parameters(), lr=0.03)
        # optimizerG = torch.optim.SGD(G.parameters(), lr=0.03)
        optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002)
        optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002)
        lab_real = torch.ones(64, 1, device=device)
        lab_fake = torch.zeros(64, 1, device=device)

        collect_lossD = []
        collect_lossG = []

        # for logging:
        collect_x_gen = []
        fixed_noise = torch.randn(64, 100, device=device)
        fig = plt.figure()  # keep updating this one
        plt.ion()

        for epoch in range(num_epoch):  # 3 epochs
            lossD_sum, lossG_sum = 0, 0
            for i, data in enumerate(dataloader, 0):
                # STEP 1: Discriminator optimization step
                x_real, _ = next(iter(dataloader))
                x_real = x_real.to(device)
                # reset accumulated gradients from previous iteration
                optimizerD.zero_grad()

                D_x = D(x_real)
                lossD_real = criterion(D_x, lab_real)

                z = torch.randn(
                    64, 100, device=device
                )  # random noise, 64 samples, z_dim=100
                x_gen = G(z).detach()
                D_G_z = D(x_gen)
                lossD_fake = criterion(D_G_z, lab_fake)

                lossD = lossD_real + lossD_fake
                lossD.backward()
                optimizerD.step()

                # STEP 2: Generator optimization step
                # reset accumulated gradients from previous iteration
                optimizerG.zero_grad()

                z = torch.randn(
                    64, 100, device=device
                )  # random noise, 64 samples, z_dim=100
                x_gen = G(z)
                D_G_z = D(x_gen)
                lossG = criterion(D_G_z, lab_real)  # -log D(G(z))

                lossG.backward()
                optimizerG.step()

                lossD_sum += lossD.item()
                lossG_sum += lossG.item()

                if i % 100 == 0:
                    x_gen = G(fixed_noise)
                    show_imgs(x_gen, new_fig=False)
                    fig.canvas.draw()
                    print(
                        "e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}".format(
                            epoch,
                            i,
                            len(dataloader),
                            D_x.mean().item(),
                            D_G_z.mean().item(),
                        )
                    )

            collect_lossD.append(lossD_sum / (i + 1))
            collect_lossG.append(lossG_sum / (i + 1))

            # End of epoch
            x_gen = G(fixed_noise)
            collect_x_gen.append(x_gen.detach().clone())

        save_imgs(collect_x_gen[-1])

        # Save figures
        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(1, num_epoch + 1), collect_lossD)
        plt.savefig("discriminative_loss.pdf", format="pdf", bbox_inches="tight")

        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(1, num_epoch + 1), collect_lossG)
        plt.savefig("generative_loss.pdf", format="pdf", bbox_inches="tight")

        # Save the model
        torch.save(
            {
                "discriminator_state_dict": D.state_dict(),
                "generator_state_dict": G.state_dict(),
            },
            "model.pth",
        )

    else:
        cnn_G = CNNGenerator()
        cnn_D = CNNDiscriminator()
        input_test = torch.rand((64, 1, 28, 28))
        d_out = cnn_D(input_test)
        print(d_out.shape)
        input_test = torch.rand(64, 100)
        g_out = cnn_G(input_test)
        print(g_out.shape)
        d_out = cnn_D(g_out)
        print(d_out.shape)
        print(cnn_D)
        print(cnn_G)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Device: ", device)
        # Re-initialize D, G:
        D = CNNDiscriminator().to(device)
        G = CNNGenerator().to(device)
        # Now let's set up the optimizers (Adam, better than SGD for this)
        # optimizerD = torch.optim.SGD(D.parameters(), lr=0.03)
        # optimizerG = torch.optim.SGD(G.parameters(), lr=0.03)
        optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002)
        optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002)
        lab_real = torch.ones(64, 1, device=device)
        lab_fake = torch.zeros(64, 1, device=device)

        collect_lossD = []
        collect_lossG = []

        # for logging:
        collect_x_gen = []
        fixed_noise = torch.randn(64, 100, device=device)
        fig = plt.figure()  # keep updating this one
        plt.ion()

        for epoch in range(num_epoch):  # 3 epochs
            lossD_sum, lossG_sum = 0, 0
            for i, data in enumerate(dataloader, 0):
                # STEP 1: Discriminator optimization step
                x_real, _ = next(iter(dataloader))
                x_real = x_real.to(device)
                # reset accumulated gradients from previous iteration
                optimizerD.zero_grad()

                D_x = D(x_real)
                lossD_real = criterion(D_x, lab_real)

                z = torch.randn(
                    64, 100, device=device
                )  # random noise, 64 samples, z_dim=100
                x_gen = G(z).detach()
                D_G_z = D(x_gen)
                lossD_fake = criterion(D_G_z, lab_fake)

                lossD = lossD_real + lossD_fake
                lossD.backward()
                optimizerD.step()

                # STEP 2: Generator optimization step
                # reset accumulated gradients from previous iteration
                optimizerG.zero_grad()

                z = torch.randn(
                    64, 100, device=device
                )  # random noise, 64 samples, z_dim=100
                x_gen = G(z)
                D_G_z = D(x_gen)
                lossG = criterion(D_G_z, lab_real)  # -log D(G(z))

                lossG.backward()
                optimizerG.step()

                lossD_sum += lossD.item()
                lossG_sum += lossG.item()

                if i % 100 == 0:
                    x_gen = G(fixed_noise)
                    show_imgs(x_gen, new_fig=False)
                    fig.canvas.draw()
                    print(
                        "e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}".format(
                            epoch,
                            i,
                            len(dataloader),
                            D_x.mean().item(),
                            D_G_z.mean().item(),
                        )
                    )

            collect_lossD.append(lossD_sum / (i + 1))
            collect_lossG.append(lossG_sum / (i + 1))

            # End of epoch
            x_gen = G(fixed_noise)
            collect_x_gen.append(x_gen.detach().clone())

        save_imgs(collect_x_gen[-1], "cnn_output.pdf")

        # Save figures
        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(1, num_epoch + 1), collect_lossD)
        plt.savefig("cnn_discriminative_loss.pdf", format="pdf", bbox_inches="tight")

        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(1, num_epoch + 1), collect_lossG)
        plt.savefig("cnn_generative_loss.pdf", format="pdf", bbox_inches="tight")

        # Save the model
        torch.save(
            {
                "discriminator_state_dict": D.state_dict(),
                "generator_state_dict": G.state_dict(),
            },
            "cnn_model.pth",
        )
else:
    # Load the model
    if not use_conv:
        D = Discriminator().to(device)
        G = Generator().to(device)
        checkpoint = torch.load("model.pth")
        D.load_state_dict(checkpoint["discriminator_state_dict"])
        G.load_state_dict(checkpoint["generator_state_dict"])

        # Define random numbers
        torch.manual_seed(42)
        custom_noise = torch.randn(8, 100, device=device)
        print(custom_noise)
        generated_images = G(custom_noise)
        save_imgs(generated_images, "generated_images.pdf")

        # Randomly select 5 noise vectors for adjustment
        selected_noise_indices = torch.tensor([20, 50, 60, 70, 90])
        adjusted_images = []

        for idx in selected_noise_indices:
            for adjustment in [2, 5, -5]:
                adjusted_noise = custom_noise.clone()
                adjusted_noise[:, idx] += adjustment
                adjusted_images.append(G(adjusted_noise))

        save_combined_imgs(adjusted_images, "adjusted_images")
    else:
        D = CNNDiscriminator().to(device)
        G = CNNGenerator().to(device)
        checkpoint = torch.load("cnn_model.pth")
        D.load_state_dict(checkpoint["discriminator_state_dict"])
        G.load_state_dict(checkpoint["generator_state_dict"])

        # Define random numbers
        torch.manual_seed(42)
        custom_noise = torch.randn(8, 100, device=device)
        print(custom_noise)
        generated_images = G(custom_noise)
        save_imgs(generated_images, "generated_images.pdf")

        # Randomly select 5 noise vectors for adjustment
        selected_noise_indices = torch.tensor([20, 50, 60, 70, 90])
        adjusted_images = []

        for idx in selected_noise_indices:
            for adjustment in [2, 5, -5]:
                adjusted_noise = custom_noise.clone()
                adjusted_noise[:, idx] += adjustment
                adjusted_images.append(G(adjusted_noise))

        save_combined_imgs(adjusted_images, "adjusted_images")
