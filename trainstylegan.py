from stylegan import StyledGenerator, Discriminator
import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset.dataset import Dataset
import jittor.transform as transform
import argparse
import numpy as np
import math, random, os
from tqdm import tqdm
import matplotlib.pyplot as plt
jt.flags.use_cuda = 1
jt.flags.log_silent = True
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].update(par1[k] * decay + (1 - decay) * par2[k].detach())

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
class MultiResolutionDataset(Dataset):
    def __init__(self, root_path, transform, resolution=8):
        super().__init__()
        resolution_path = os.path.join(root_path, str(resolution))
        train_image = []
        for image_file in os.listdir(resolution_path):
            image_path = os.path.join(resolution_path, image_file)
            ext = os.path.splitext(image_path)[-1]
            
            if ext not in ['.png', '.jpg']:
                continue
                
            image = plt.imread(image_path)
            
            if ext == '.png':
                image = image * 255
                
            image = image.astype('uint8')
            train_image.append(image)
        self.train_image = train_image
        self.transform  = transform
        self.resolution = resolution
        
    def __len__(self):
        return len(self.train_image)
    
    def __getitem__(self, index):
        X = self.train_image[index]
        return self.transform(X)
parser = argparse.ArgumentParser()
parser.add_argument('--maxiter', type=int, default=40000, help='number of epochs of training')
parser.add_argument('--init_size', default=8, type=int, help='initial image size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--max_size', type=int, default=128, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--image_dir',default='image',help='image directory')
parser.add_argument('--opt_dir',default='image/gan')
parser.add_argument('--model_dir',default='image/gan')
parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
parser.add_argument('--phase', type=int, default=15_000, help='number of samples used for each training phases',)
parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
parser.add_argument('--loss',type=str,default='r1',choices=['wgan-gp', 'r1'],help='class of gan loss',)    
args = parser.parse_args()
code_size = 512
# dataset
transform = transform.Compose([
    transform.ToPILImage(),
    transform.RandomHorizontalFlip(),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# model
generator = StyledGenerator(code_dim=code_size)
discriminator = Discriminator(from_rgb_activate=True)
g_running = StyledGenerator(code_size)
g_running.eval()
d_optimizer = jt.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
g_optimizer = jt.optim.Adam(generator.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
g_optimizer.add_param_group({
    'params': generator.style.parameters(),
    'lr': args.lr * 0.01,
    'mult': 0.01,
    }
)
accumulate(g_running, generator, 0)
if args.ckpt is not None:
    ckpt = jt.load(args.ckpt)

    generator.load_state_dict(ckpt['generator'])
    discriminator.load_state_dict(ckpt['discriminator'])
    g_running.load_state_dict(ckpt['g_running'])

    print('resuming from checkpoint .......')

# use lr scheduling
args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}
args.gen_sample = {512: (8, 4), 1024: (4, 2)}
args.batch_default = 32

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult
def train(args, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = MultiResolutionDataset(args.image_dir,transform=transform, resolution=resolution).set_attrs(batch_size=args.batch.get(resolution, args.batch_default), shuffle=True)
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(args.maxiter))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:
        # discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = MultiResolutionDataset(args.image_dir,transform=transform, resolution=resolution).set_attrs(batch_size=args.batch.get(resolution, args.batch_default))
            data_loader = iter(loader)

            jt.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'{args.model_dir}/checkpoint/train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            # (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = nn.softplus(-real_scores).mean()
            # real_predict.backward(retain_graph=True)

            grad_real = jt.grad(real_scores.sum(), real_image)
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            # grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = jt.randn(
                4, b_size, code_size
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = jt.randn(2, b_size, code_size).chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            # fake_predict.backward()

            eps = jt.rand(b_size, 1, 1, 1)
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = jt.grad(hat_predict.sum(), x_hat)
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            # grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = jt.nn.softplus(fake_predict).mean()
            # fake_predict.backward()
            if i%10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()
        loss_D = real_predict + grad_penalty + fake_predict
        d_optimizer.step(loss_D)

        
        # generator.zero_grad()

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_image = generator(gen_in2, step=step, alpha=alpha)

        predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            loss = -predict.mean()

        elif args.loss == 'r1':
            loss = nn.softplus(-predict).mean()

        if i%10 == 0:
            gen_loss_val = loss.item()

        # loss.backward()
        g_optimizer.step(loss)
        accumulate(g_running, generator)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = (10, 5)

            with jt.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            jt.randn(gen_j, code_size), step=step, alpha=alpha
                        ).data
                    )

            jt.save_image(
                jt.concat(images, 0),
                f'{args.opt_dir}/sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            jt.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                }, f'{args.model_dir}/checkpoint/{str(i + 1).zfill(6)}.model'
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)
train(args, generator, discriminator)