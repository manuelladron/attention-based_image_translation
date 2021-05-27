import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import copy
import sys

from torchvision.models import vgg19

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print('device: ', device)

class PerceptualLoss(nn.Module):
    """Called by Criterion"""

    def __init__(self, layers):
        super().__init__()
        self.cnn = vgg19(pretrained=True).features.to(device).eval()
        self.layers = layers

    def get_model(self, content_img):
        model, _, content_losses = get_model_and_losses(self.cnn, content_img,
                                                        content_layers=self.layers)

        return model, content_losses


class Criterion(nn.Module):
    #     def __init__(self, args, mask=False, layer=['conv_2_1', 'conv_2_2', 'conv_3_1', 'conv_3_2']):
    #     def __init__(self, args, mask=False, layer=['conv_1_1','conv_1_2','conv_2_1']):
    def __init__(self, args, mask=False, layer=['conv_1_1', 'conv_1_2', 'conv_2_1', 'conv_2_2', 'conv_3_1']):
        super().__init__()

        # Mask
        self.mask = mask

        # Perceptual loss from layers in layers
        self.layer = layer
        self.perceptual = PerceptualLoss(layer)

        # Weights
        self.pixel_loss_w = args.pixel_loss_wgt
        self.perc_w = args.perc_wgt

    def forward(self, pred, target):  # pred and target are of shape [1, 3, 64, 64]
        """Calculate loss of prediction and target. in p-norm / perceptual  space"""

        number_losses = len(self.perceptual.layers)
        # print('num losses: ', number_losses)
        if self.mask:
            target, mask = target
            # todo: loss with mask
            pixel_loss = F.mse_loss(mask * pred, mask * target)
            # loss = self.perc(pred, target, mask)
            perc_model, perc_losses = self.perceptual.get_model(mask * target)
            perc_model(mask * pred)
            perc_loss = 0
            for pl in perc_losses:
                perc_loss += pl.loss

        else:
            # todo: loss w/o mask
            pixel_loss = F.mse_loss(pred, target)  # it's in the 0.something range
            perc_model, perc_losses = self.perceptual.get_model(target)  # in the 14 and 80
            # print('forward pass')
            perc_model(pred)
            perc_loss = 0
            for pl in perc_losses:
                perc_loss += pl.loss

        # Weight losses
        perc_loss *= self.perc_w
        pixel_loss *= self.pixel_loss_w

        # Add and return
        total_loss = perc_loss + pixel_loss

        return total_loss

class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # you need to `detach' the target content from the graph used to
        # compute the gradient in the forward pass that made it so that we don't track
        # those gradients anymore
        self.target = target.detach()

    def forward(self, input):
        # this needs to be a passthrough where you save the appropriate loss value
        self.loss = F.mse_loss(input, self.target)
        return input

def get_model_and_losses(cnn, content_img,
                         content_layers=None):
    cnn = copy.deepcopy(cnn)

    # Normalization
    normalization = Normalization()

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    model = nn.Sequential(normalization)

    i = 0  # everytime we see a conv we increment this 15 CONV LAYERS IN TOTAL
    block = 1
    conv_b = 1

    for layer in cnn.children():
        # print(f'i:{i}, layer: {layer.__class__.__name__}')
        if isinstance(layer, nn.Conv2d):
            i += 1
            # name = 'conv_{}'.format(i)
            name = f'conv_{block}_{conv_b}'
            conv_b += 1
        elif isinstance(layer, nn.ReLU):
            name = 'relu_'.format(i)
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            block += 1
            conv_b = 1
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)

        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        # print('\nName: ', name)

        if name in content_layers:
            target = model(content_img).detach()  # Target is our content image
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i), content_loss)
            content_losses.append(content_loss)

    # print('\nnew model: ', model)
    # chop off the layers after last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
class Normalization(nn.Module):
    def __init__(self, mean=cnn_normalization_mean, std=cnn_normalization_std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().view(-1,1,1)
        self.std = std.clone().view(-1,1,1)
#         self.mean = torch.tensor(mean).view(-1,1,1)
#         self.std = torch.tensor(std).view(-1,1,1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std