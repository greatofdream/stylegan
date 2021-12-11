import jittor as jt
from jittor import nn, Module

class Generator(Module):
    def __init__(self, dim=3):
        super(Generator, self).__init__()
        self.fc = nn.Linear(1024, 8*8*256)
        self.fc_bn = nn.BatchNorm(256)
        self.deconv1 = nn.ConvTranspose(256, 256, 3, 2, 1, 1)# L*2
        self.deconv1_bn = nn.BatchNorm(256)
        self.deconv2 = nn.ConvTranspose(256, 256, 3, 1, 1)# L*1
        self.deconv2_bn = nn.BatchNorm(256)
        self.deconv3 = nn.ConvTranspose(256, 256, 3, 2, 1, 1)
        self.deconv3_bn = nn.BatchNorm(256)
        self.deconv4 = nn.ConvTranspose(256, 256, 3, 1, 1)
        self.deconv4_bn = nn.BatchNorm(256)
        self.deconv5 = nn.ConvTranspose(256, 128, 3, 2, 1, 1)
        self.deconv5_bn = nn.BatchNorm(128)
        self.deconv6 = nn.ConvTranspose(128, 64, 3, 2, 1, 1)
        self.deconv6_bn = nn.BatchNorm(64)
        self.deconv7 = nn.ConvTranspose(64 , dim, 3, 1, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def execute(self, input):
        x = self.fc(input).reshape((-1, 256, 8, 8))
        x = self.relu(self.fc_bn(x))
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        x = self.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.relu(self.deconv4_bn(self.deconv4(x)))
        x = self.relu(self.deconv5_bn(self.deconv5(x)))
        x = self.relu(self.deconv6_bn(self.deconv6(x)))
        x = self.tanh(self.deconv7(x))
        return x
 
class Discriminator(nn.Module):
    def __init__(self, dim=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv(dim, 64, 5, 2, 2)#(L+1)/2
        self.conv2 = nn.Conv(64, 128, 5, 2, 2)
        self.conv2_bn = nn.BatchNorm(128)
        self.conv3 = nn.Conv(128, 256, 5, 2, 2)
        self.conv3_bn = nn.BatchNorm(256)
        self.conv4 = nn.Conv(256, 512, 5, 2, 2)
        self.conv4_bn = nn.BatchNorm(512)
        self.fc = nn.Linear(512*8*8, 1)
        self.leaky_relu = nn.Leaky_relu()

    def execute(self, input):
        x = self.leaky_relu(self.conv1(input), 0.2)
        x = self.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = self.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = x.reshape((x.shape[0], 512*8*8))
        x = self.fc(x)
        return x

def Loss(x, b):
    mini_batch = x.shape[0]
    y_real_ = jt.ones((mini_batch,))
    y_fake_ = jt.zeros((mini_batch,))
    if b:
        return (x-y_real_).sqr().mean()
    else:
        return (x-y_fake_).sqr().mean()