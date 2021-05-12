# CMU 16-726 Learning-Based Image Synthesis / Spring 2021, Final Project

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from ..sagan.spectral import *


class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        # Construct the conv layers
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B * N * C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B * C * N
        energy = torch.bmm(proj_query, proj_key)  # batch matrix-matrix product

        attention = self.softmax(energy)  # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B * C * N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # batch matrix-matrix product
        out = out.view(m_batchsize, C, width, height)  # B * C * W * H

        # Add attention weights onto input
        out = self.gamma * out + x
        return out, attention


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='batch'):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='batch', init_zero_weights=False, spectral=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    if spectral:
        print('---- SPECTRAL NORMALIZATION ------')
        conv_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
    else:
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(DCGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        self.deconv1 = deconv(100, 256, 4, stride=1, padding=0, norm='instance')
        self.deconv2 = deconv(256, 128, 4, stride=2, padding=1, norm='instance')
        self.deconv3 = deconv(128, 64, 4, stride=2, padding=1, norm='instance')
        self.deconv4 = deconv(64, 32, 4, stride=2, padding=1, norm='instance')
        self.deconv5 = deconv(32, 3, 4, stride=2, padding=1, norm=None)

    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1

            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x32x32
        """

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        out = F.relu(self.deconv1(z))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = F.tanh(self.deconv5(out))

        return out


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim, norm):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out



class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False, norm='batch'):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, norm=norm)
        self.conv2 = conv(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, norm=norm)

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim=64, norm=norm)
        self.resnet_block2 = ResnetBlock(conv_dim=64, norm=norm)
        self.resnet_block3 = ResnetBlock(conv_dim=64, norm=norm)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv1 = deconv(64, 32, 4, stride=2, padding=1, norm='instance')
        self.deconv2 = deconv(32, 3, 4, stride=2, padding=1, norm='instance')

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block(out))
        out = F.relu(self.resnet_block2(out))
        out = F.relu(self.resnet_block3(out))

        out = F.relu(self.deconv1(out))
        out = F.tanh(self.deconv2(out))

        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, norm='batch', spectral=False):
        super(DCDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        self.conv1 = conv(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, norm='instance',
                          init_zero_weights=False, spectral=spectral)
        self.conv2 = conv(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, norm='instance',
                          init_zero_weights=False, spectral=spectral)
        self.conv3 = conv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, norm='instance',
                          init_zero_weights=False, spectral=spectral)
        self.conv4 = conv(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, norm='instance',
                          init_zero_weights=False, spectral=spectral)
        self.conv5 = conv(in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=0, norm=None,
                          init_zero_weights=False, spectral=spectral)

    def forward(self, x):
        out = F.relu(self.conv1(x))

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out).squeeze()

        return out

    
class PatchDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, ndf=64, norm='batch', spectral=False):
        super(PatchDiscriminator, self).__init__()

        # First layer does not have norm layer
        self.conv1 = conv(in_channels=3, out_channels=ndf, kernel_size=4, stride=2, padding=1, norm=None,
                          init_zero_weights=False, spectral=spectral)
        
        self.conv2 = conv(in_channels=ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, norm='instance',
                          init_zero_weights=False, spectral=spectral)
        self.conv3 = conv(in_channels=ndf*2, out_channels=ndf*4, kernel_size=4, stride=2, padding=1, norm='instance',
                          init_zero_weights=False, spectral=spectral)
        self.conv4 = conv(in_channels=ndf*4, out_channels=ndf*8, kernel_size=4, stride=2, padding=1, norm='instance',
                          init_zero_weights=False, spectral=spectral)
        
        # Last layer does not have norm and has stride 1 
        self.conv5 = conv(in_channels=ndf*8, out_channels=1, kernel_size=4, stride=2, padding=0, norm=None,
                          init_zero_weights=False, spectral=spectral)

    def forward(self, x):
        #print('PatchGAN')
        out = F.relu(self.conv1(x))
        #print('1st layer: ', out.shape)
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        out = F.leaky_relu(self.conv2(out), 0.2, True)
        #print('2nd layer: ', out.shape)
        out = F.leaky_relu(self.conv3(out), 0.2, True)
        #print('3rd layer: ', out.shape)
        out = F.leaky_relu(self.conv4(out), 0.2, True)
        #print('4th layer: ', out.shape)
        out = self.conv5(out).squeeze()
        #print('5th layer: ', out.shape)

        return out 
    

def weights_init2(m, init_='xavier'):
    classname = m.__class__.__name__
    print('Initializing weights')
    if init_ != 'xavier' or init_ != 'glorot':
        print('\n ----------------------- Initializing Normal -----------------------')
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        
    else:
        print('\n ----------------------- Initializing Xavier -----------------------')
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.xavier_uniform_(m.weight.data)
            
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)