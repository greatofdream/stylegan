#!/usr/bin/python3
# copy from https://github.com/Jittor/gan-jittor/models/gan/gan.py
import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset.dataset import ImageFolder
# from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import time
import cv2
import matplotlib.pyplot as plt

jt.flags.use_cuda = 1
#os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--image_dir',default='image',help='image directory')
parser.add_argument('--opt_dir',default='image/gan')
parser.add_argument('--model', default='gan')
parser.add_argument('--model_dir', default=None)
opt = parser.parse_args()
print(opt)
from lsgan import Generator, Discriminator, Loss
img_shape = (opt.channels, opt.img_size, opt.img_size)
outputpath = opt.opt_dir

# Configure data loader
transform = transform.Compose([
    transform.Resize(size=opt.img_size),
    # transform.Gray(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
dataloader = ImageFolder(opt.image_dir).set_attrs(transform=transform, batch_size=opt.batch_size, shuffle=True)

def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    img2=img.reshape([-1,W*nrow*nrow,H])
    img=img2[:,:W*nrow,:]
    for i in range(1,nrow):
        img=np.concatenate([img,img2[:,W*nrow*i:W*nrow*(i+1),:]],axis=2)
    img=(img+1.0)/2.0*255
    img=img.transpose((1,2,0))
    cv2.imwrite(path,img)
adversarial_loss = Loss

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
if opt.model_dir is not None:
    generator.load('{}/G.pkl'.format(opt.model_dir))
    discriminator.load('{}/D.pkl'.format(opt.model_dir))
# Optimizers
# optimizer_G = Optimizer_G()#jt.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = Optimizer_D()#jt.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# 学习率
lr = 0.0002 
betas = (0.5,0.999)
optimizer_G = nn.Adam(generator.parameters(), lr, betas=betas)
# 判别器优化器
optimizer_D = nn.Adam(discriminator.parameters(), lr, betas=betas)

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------
def train(epoch):
    for batch_idx, (x_, target) in enumerate(dataloader):
        mini_batch = x_.shape[0]
        # 判别器训练
        D_result = discriminator(x_)
        D_real_loss = Loss(D_result, True)
        z_ = init.gauss((mini_batch, 1024), 'float')
        G_result = generator(z_)
        D_result_ = discriminator(G_result)
        D_fake_loss = Loss(D_result_, False)
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.sync()
        optimizer_D.step(D_train_loss)

        # 生成器训练
        z_ = init.gauss((mini_batch, 1024), 'float')
        G_result = generator(z_)
        D_result = discriminator(G_result)
        G_train_loss = Loss(D_result, True)
        G_train_loss.sync()
        optimizer_G.step(G_train_loss)
        if (batch_idx%100==0):
            print('D training loss =', D_train_loss.data.mean())
            print('G training loss =', G_train_loss.data.mean())

# def validate(epoch):
#     D_losses = []
#     G_losses = []
#     generator.eval()
#     discriminator.eval()
#     for batch_idx, (x_, target) in enumerate(eval_loader):
#         mini_batch = x_.shape[0]
#         # 判别器损失计算
#         D_result = discriminator(x_)
#         D_real_loss = Loss(D_result, True)
#         z_ = jt.init.gauss((mini_batch, 1024), 'float')
#         G_result = generator(z_)
#         D_result_ = discriminator(G_result)
#         D_fake_loss = Loss(D_result_, False)
#         D_train_loss = D_real_loss + D_fake_loss
#         D_losses.append(D_train_loss.data.mean())

#         # 生成器损失计算
#         z_ = jt.init.gauss((mini_batch, 1024), 'float')
#         G_result = generator(z_)
#         D_result = discriminator(G_result)
#         G_train_loss = Loss(D_result, True)
#         G_losses.append(G_train_loss.data.mean())
#     generator.train()
#     discriminator.train()
#     print("validate: epoch",epoch)
#     print('  D validate loss =', np.array(D_losses).mean())
#     print('  G validate loss =', np.array(G_losses).mean())
fixed_z_ = jt.init.gauss((5 * 5, 1024), 'float')
def save_result(num_epoch, G , path = 'result.png'):
    """Use the current generator to generate 5*5 pictures and store them.
    Args:
        num_epoch(int): current epoch
        G(generator): current generator
        path(string): storage path of result image
    """

    z_ = fixed_z_
    G.eval()
    test_images = G(z_)
    G.train()
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i in range(size_figure_grid):
        for j in range(size_figure_grid):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].data.transpose(1, 2, 0)+1)/2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()
for epoch in range(opt.n_epochs):
    train(epoch)
    # validate(epoch)
    save_result(epoch, generator,path='{}/{}.png'.format(outputpath, epoch))
generator.save('{}/G.pkl'.format(outputpath))
discriminator.save('{}/D.pkl'.format(outputpath))
