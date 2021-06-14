import os
import numpy as np
import math
import itertools
import datetime
import time
import sys
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import Generator_Resnet, Discriminator, weights_init_normal
from dataset import ImageDataset
from utils import LambdaLR, ReplayBuffer, opt, save_pickle, plot_losses
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc

loss_dict = {"D":[], "G":[], "A":[], "I":[], "C":[]}

cuda = torch.cuda.is_available()

# os.makedirs("/content/drive/MyDrive/Night2Day/images", exist_ok=True)
# os.makedirs("/content/drive/MyDrive/Night2Day/checkpoints", exist_ok=True)

# losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

input_shape = (3, 256, 256)
# input_shape = (3, 224, 224)

G_AB = Generator_Resnet(input_shape, 5)
G_BA = Generator_Resnet(input_shape, 5)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    Dis_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=5e-4, betas=(0.999, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=5e-4, betas=(0.999, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=5e-4, betas=(0.999, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(40, 10, 15).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(40, 10, 15).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(40, 10, 15).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

dataloader = DataLoader(ImageDataset(mode="train"), batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(ImageDataset(mode="valid"), batch_size=8, shuffle=True, num_workers=1)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arrange images along x-axis
    real_A = make_grid(real_A[:3,:,:,:], nrow=3, normalize=True)
    real_B = make_grid(real_B[:3,:,:,:], nrow=3, normalize=True)
    fake_A = make_grid(fake_A[:3,:,:,:], nrow=3, normalize=True)
    fake_B = make_grid(fake_B[:3,:,:,:], nrow=3, normalize=True)
    # Arrange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "/content/drive/MyDrive/Night2Day/images/generated_4/%s.png" % (batches_done), normalize=False)


log_file = open("/content/drive/MyDrive/Night2Day/cyclegan/cyclegan_4.log", "w+", encoding="utf-8")

prev_time = time.time()
print("Training Stated")
for epoch in range(opt.epoch, opt.n_epochs+1):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape[:1], D_A.output_shape[1]+2, D_A.output_shape[2]+2))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape[:1], D_A.output_shape[1]+2, D_A.output_shape[2]+2))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        BA = G_BA(real_A)
        AB = G_AB(real_B)
        loss_id_A = criterion_identity(BA, real_A)
        loss_id_B = criterion_identity(AB, real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        del BA, AB

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        loss_cycle_A = criterion_cycle(G_BA(fake_B), real_A)
        loss_cycle_B = criterion_cycle(G_AB(fake_A), real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        gc.collect()
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if batches_done % 250 == 0:
            loss_dict["D"].append(loss_D.item())
            loss_dict["G"].append(loss_G.item())
            loss_dict["A"].append(loss_GAN.item())
            loss_dict["I"].append(loss_identity.item())
            loss_dict["C"].append(loss_cycle.item())
        log_file.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s\n"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )
        

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s\n"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if epoch % opt.checkpoint_interval == 0 or epoch == opt.n_epochs+1:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "/content/drive/MyDrive/Night2Day/checkpoints_4/G_AB_%d.pth" % (epoch))
        torch.save(G_BA.state_dict(), "/content/drive/MyDrive/Night2Day/checkpoints_4/G_BA_%d.pth" % (epoch))
        torch.save(D_A.state_dict(), "/content/drive/MyDrive/Night2Day/checkpoints_4/D_A_%d.pth" % (epoch))
        torch.save(D_B.state_dict(), "/content/drive/MyDrive/Night2Day/checkpoints_4/D_B_%d.pth" % (epoch))
save_pickle("/content/drive/MyDrive/Night2Day/cyclegan/pickles/losses.dump.pkl", loss_dict)
try:
    plot_losses(loss_dict)
except Exception as e:
    print("Cannot plot the losses")