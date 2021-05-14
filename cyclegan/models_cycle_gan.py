# CMU 16-726 Learning-Based Image Synthesis / Spring 2021, Final Project

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from spectral import *
import time
from skimage.transform import rescale, resize, downscale_local_mean
import pdb
import math
import numbers
from blurpool import BlurPool
torch.set_printoptions(edgeitems=15)


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)




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
                output does not change the input shape
        """
        # print('----SELF ATTENTION------')
        # print('input: ', x.shape)
        m_batchsize, C, width, height = x.size()

        # print('batch: ', m_batchsize)
        # print('C: ', C)
        # print('width: ', width)
        # print('height: ', height)

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B * N * C
        #print('proj query: ', proj_query.shape)
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


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, bias=False, num_heads=num_heads)

    def forward(self, x):
        return self.mha(x, x, x)


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, patch_dim, num_heads=8):
        super(ViTBlock, self).__init__()
        self.ln1 = nn.LayerNorm([embed_dim, patch_dim])
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, bias=False, num_heads=num_heads)
        self.ln2 = nn.LayerNorm([embed_dim, patch_dim])
        self.gelu = nn.GELU()

        # using conv1d with kernel_size=1 is like applying a linear layer to the channel dim
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=False)

    def forward(self, x):
        skip = x
        x = self.ln1(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.mha(x, x, x, need_weights=False)
        x = torch.transpose(x, 1, 2)
        x = x + skip
        skip = x
        x = self.ln2(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = x + skip
        return x


class MixerBlock(nn.Module):
    def __init__(self, embed_dim, patch_dim):
        super(MixerBlock, self).__init__()
        self.ln1 = nn.LayerNorm([embed_dim, patch_dim])

        self.dense1 = nn.Linear(in_features=patch_dim, out_features=patch_dim, bias=False)
        self.gelu1 = nn.GELU()
        self.dense2 = nn.Linear(in_features=patch_dim, out_features=patch_dim, bias=False)

        self.ln2 = nn.LayerNorm([embed_dim, patch_dim])

        # using conv1d with kernel_size=1 is like applying a linear layer to the channel dim
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=False)
        self.gelu2 = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=False)

    def forward(self, x):
        skip = x
        x = self.ln1(x)
        x = self.dense1(x)
        x = self.gelu1(x)
        x = self.dense2(x)
        x = x + skip
        skip = x
        x = self.ln2(x)
        x = self.conv1(x)
        x = self.gelu2(x)
        x = self.conv2(x)
        x = x + skip
        return x


class CycleGeneratorViT(nn.Module):

    def __init__(self, patch_dim, embed_dim=256, transform_layers=4, patch_size=8, num_heads=8):
        super(CycleGeneratorViT, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim // 4, kernel_size=4, stride=2, padding=1)
        self.in_c1 = nn.InstanceNorm2d(embed_dim // 4)
        self.conv2 = nn.Conv2d(in_channels=embed_dim // 4, out_channels=embed_dim // 2, kernel_size=4, stride=2,
                               padding=1)
        self.in_c2 = nn.InstanceNorm2d(embed_dim // 2)

        self.conv_down = nn.Conv2d(in_channels=embed_dim // 2, out_channels=embed_dim, kernel_size=patch_size,
                                   stride=patch_size)

        self.blocks = [ViTBlock(embed_dim=embed_dim, patch_dim=patch_dim, num_heads=num_heads) for _ in range(transform_layers)]
        self.blocks = nn.ModuleList(self.blocks)

        self.deconv_up = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=patch_size,
                                            stride=patch_size)

        self.deconv_1 = nn.ConvTranspose2d(in_channels=embed_dim // 2, out_channels=embed_dim // 4, kernel_size=4,
                                           stride=2, padding=1)
        self.in_d1 = nn.InstanceNorm2d(embed_dim // 4)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=embed_dim // 4, out_channels=embed_dim // 4, kernel_size=4,
                                           stride=2, padding=1)
        self.in_d2 = nn.InstanceNorm2d(embed_dim // 4)

        self.conv_out = nn.Conv2d(in_channels=embed_dim // 4, out_channels=3, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        out = x

        out = self.conv1(out)
        out = self.in_c1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.in_c2(out)
        out = F.relu(out)

        out = self.conv_down(out)
        patch_h, patch_w = out.shape[2], out.shape[3]
        out = out.view(out.shape[0], out.shape[1], -1)
        for b in self.blocks:
            out = b(out)
        out = out.view(out.shape[0], out.shape[1], patch_h, patch_w)
        out = self.deconv_up(out)
        out = F.relu(out)

        out = self.deconv_1(out)
        out = self.in_d1(out)
        out = F.relu(out)

        out = self.deconv_2(out)
        out = self.in_d2(out)
        out = F.relu(out)

        out = self.conv_out(out)

        out = F.tanh(out)
        return out


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out



class CycleGeneratorMixer(nn.Module):

    def __init__(self, patch_dim, image_size=256, embed_dim=256, transform_layers=4, patch_size=8, num_heads=8):
        super(CycleGeneratorMixer, self).__init__()

        # stem
        model = [
            nn.Conv2d(in_channels=3, out_channels=embed_dim//4, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dim//4),
            nn.ReLU(True),
        ]

        # downsampling
        model += [
            nn.Conv2d(in_channels=embed_dim//4, out_channels=embed_dim//2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(embed_dim//2),
            nn.ReLU(True),

            nn.Conv2d(in_channels=embed_dim//2, out_channels=embed_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(True),
        ]

        # linear projection
        model += [nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)]

        # reshape
        model += [View((-1, embed_dim, patch_dim))]

        # transformation
        model += [MixerBlock(embed_dim=embed_dim, patch_dim=patch_dim) for _ in range(transform_layers)]

        model += [View((-1, embed_dim, image_size//4//patch_size, image_size//4//patch_size))]

        # linear de-projection
        model += [nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)]

        # upsampling
        model += [
            nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim//2, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(embed_dim//2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=embed_dim//2, out_channels=embed_dim//4, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(embed_dim//4),
            nn.ReLU(True),
        ]

        model += [nn.Conv2d(in_channels=embed_dim//4, out_channels=3, kernel_size=7, padding=3, padding_mode='reflect')]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):

        return self.model(x)


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
        self.g1 = BlurPool(32)
        self.a1 = Self_Attn(32)
        self.gd1 = deconv(32, 32, 4, stride=2, padding=1, norm='instance')

        self.conv2 = conv(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, norm=norm)
        self.g2 = BlurPool(64)
        self.a2 = Self_Attn(64)
        self.gd2 = deconv(64, 64, 4, stride=2, padding=1, norm='instance')

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim=64, norm=norm)
        self.gb1 = BlurPool(64)
        self.ab1 = Self_Attn(64)
        self.gdb1 = deconv(64, 64, 4, stride=2, padding=1, norm='instance')

        self.resnet_block2 = ResnetBlock(conv_dim=64, norm=norm)
        self.gb2 = BlurPool(64)
        self.ab2 = Self_Attn(64)
        self.gdb2 = deconv(64, 64, 4, stride=2, padding=1, norm='instance')

        self.resnet_block3 = ResnetBlock(conv_dim=64, norm=norm)
        self.gb3 = BlurPool(64)
        self.ab3 = Self_Attn(64)
        self.gdb3 = deconv(64, 64, 4, stride=2, padding=1, norm='instance')

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv1 = deconv(64, 32, 4, stride=2, padding=1, norm='instance')
        self.g3 = BlurPool(32)
        self.ad1 = Self_Attn(32)
        self.gd3 = deconv(32, 32, 4, stride=2, padding=1, norm='instance')

        self.deconv2 = deconv(32, 3, 4, stride=2, padding=1, norm='instance')
        self.g4 = BlurPool(3)
        self.ad2 = Self_Attn(3)
        self.gd4 = deconv(3, 3, 4, stride=2, padding=1, norm='instance')


    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 64 x 64

            Output
            ------
                out: BS x 3 x 64 x 64
        """
        # print('----CycleGan Generator Forward----')
        # print('input: ', x.shape)
        st = time.time()
        print('input: ', x.shape)
        out = F.relu(self.conv1(x))    # [BATCH, C=32, W=32, H=32]
        print('1st conv time: ', time.time()-st)
        st_at = time.time()
        out = self.g1(out)
        print('shape after g1: ', out.shape)
        out, att = self.a1(out)
        print('shape after a1: ', out.shape)
        out = self.gd1(out)
        print('shape after gd1: ', out.shape)

        print('\nAtt 1st conv time: ', time.time() - st_at)
        #print('att shape: ', att.shape)
        #pdb.set_trace()
        #print('1st conv layer: ', out.shape)
        out = F.relu(self.conv2(out))  # [BATCH, C=64, W=16, H=16]
        print('2nd conv shape: ', out.shape)
        out = self.g2(out)
        print('g2: ', out.shape)
        out, att = self.a2(out)
        print('att2 shape: ', att.shape)
        print('a2 shape: ', out.shape)
        out = self.gd2(out)
        print('gd2 shape: ', out.shape)

        #print('2nd conv layer: ', out.shape)
        out = F.relu(self.resnet_block(out))  # [BATCH, C=64, W=16, H=16]
        print('res1: ', out.shape)
        out = self.gb1(out)
        print('gb1: ', out.shape)
        out, att = self.ab1(out)
        print('att shape: ', att.shape)
        print('att res1: ', out.shape)
        out = self.gdb1(out)
        print('deconv res1: ', out.shape)

        #print('resnet block 1 layer: ', out.shape)
        out = F.relu(self.resnet_block2(out)) # [BATCH, C=64, W=16, H=16]
        print('res2: ', out.shape)
        out = self.gb2(out)
        print('gb2: ', out.shape)
        out, att = self.ab2(out)
        print('att res2: ', out.shape)
        out = self.gdb2(out)
        print('deconv res2: ', out.shape)


        #print('resnet block 2 layer: ', out.shape)
        out = F.relu(self.resnet_block3(out)) # [BATCH, C=64, W=16, H=16]
        print('resnet block 3 layer: ', out.shape)
        out = self.gb3(out)
        print('gb3: ', out.shape)
        out, att = self.ab3(out)
        print('att res3: ', out.shape)
        out = self.gdb3(out)
        print('deconv res3: ', out.shape)


        out = F.relu(self.deconv1(out))       # [BATCH, C=32, W=32, H=32]
        print('1st deconv layer: ', out.shape)
        out = self.g3(out)
        print('g3: ', out.shape)
        out, att = self.ad1(out)
        print('att deconv1: ', out.shape)
        out = self.gd3(out)
        print('deconv deconv1: ', out.shape)

        out = F.tanh(self.deconv2(out))       # [BATCH, C=3, W=64, H=64]
        print('last deconv layer: ', out.shape)
        out = self.g4(out)
        print('g4: ', out.shape)
        out, att = self.ad2(out)
        print('att deconv2: ', out.shape)
        out = self.gd4(out)
        print('deconv deconv2: ', out.shape)

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
        self.conv4 = conv(in_channels=ndf*4, out_channels=ndf*8, kernel_size=4, stride=1, padding=1, norm='instance',
                          init_zero_weights=False, spectral=spectral)
        
        # Last layer does not have norm and has stride 1 
        self.conv5 = conv(in_channels=ndf*8, out_channels=1, kernel_size=4, stride=1, padding=1, norm=None,
                          init_zero_weights=False, spectral=spectral)

    def forward(self, x):                                 # input [batch, 3, 64, 64]
        # print('PatchGAN')
        out = F.relu(self.conv1(x))                       # [batch, 64, 32, 32]
        #print('1st layer: ', out.shape)
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        out = F.leaky_relu(self.conv2(out), 0.2, True)    # [batch, 128, 16, 16]
        #print('2nd layer: ', out.shape)
        out = F.leaky_relu(self.conv3(out), 0.2, True)    # [batch, 256, 8, 8]
        #print('3rd layer: ', out.shape)
        out = F.leaky_relu(self.conv4(out), 0.2, True)    # [batch, 512, 4, 4]
        #print('4th layer: ', out.shape)
        out = self.conv5(out).squeeze()                   # [batch]
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