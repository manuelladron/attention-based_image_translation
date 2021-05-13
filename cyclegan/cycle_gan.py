# CMU 16-726 Learning-Based Image Synthesis / Spring 2021, Assignment 3
#
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# This is the main training file for the second part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to samples_cyclegan/):
#       python cycle_gan.py
#
#    To train with cycle consistency loss (saves results to samples_cyclegan_cycle/):
#       python cycle_gan.py --use_cycle_consistency_loss
#
#
#    For optional experimentation:
#    -----------------------------
#    If you have a powerful computer (ideally with a GPU), then you can obtain better results by
#    increasing the number of filters used in the generator and/or discriminator, as follows:
#      python cycle_gan.py --g_conv_dim=64 --d_conv_dim=64

import argparse
import os
import warnings

# Matplotlib
import matplotlib.pyplot as plt

import imageio
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.optim as optim

# Numpy & Scipy imports
import numpy as np

# Local imports
import utils
from data_loader import get_data_loader
from models_cycle_gan import CycleGenerator, DCDiscriminator, PatchDiscriminator, weights_init
from models_cycle_gan import CycleGeneratorViT, CycleGeneratorMixer
from diff_aug import DiffAugment

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    print("                 G_XtoY                ")
    print("---------------------------------------")
    print(G_XtoY)
    print("---------------------------------------")

    print("                 G_YtoX                ")
    print("---------------------------------------")
    print(G_YtoX)
    print("---------------------------------------")

    print("                  D_X                  ")
    print("---------------------------------------")
    print(D_X)
    print("---------------------------------------")

    print("                  D_Y                  ")
    print("---------------------------------------")
    print(D_Y)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    model_dict = {'cycle': CycleGenerator,
                  'vit': CycleGeneratorViT,
                  'mix': CycleGeneratorMixer}

    if opts.gen=="cycle":
        G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights, norm=opts.norm)
        G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights, norm=opts.norm)

    elif opts.gen=="vit":
        G_XtoY = CycleGeneratorViT(embed_dim=opts.g_conv_dim, patch_dim=opts.patch_dim, num_heads=8, transform_layers=opts.blocks, patch_size=opts.patch)
        G_YtoX = CycleGeneratorViT(embed_dim=opts.g_conv_dim, patch_dim=opts.patch_dim, num_heads=8, transform_layers=opts.blocks, patch_size=opts.patch)

    elif opts.gen=="mix":
        G_XtoY = CycleGeneratorMixer(embed_dim=opts.g_conv_dim, patch_dim=opts.patch_dim, transform_layers=opts.blocks, patch_size=opts.patch)
        G_YtoX = CycleGeneratorMixer(embed_dim=opts.g_conv_dim, patch_dim=opts.patch_dim, transform_layers=opts.blocks, patch_size=opts.patch)


    model_dict = {'dc': DCDiscriminator}
    if opts.patch_disc:
        model_dict = {'dc': PatchDiscriminator}
        
    D_X = model_dict[opts.disc](conv_dim=opts.d_conv_dim, norm=opts.norm)
    D_Y = model_dict[opts.disc](conv_dim=opts.d_conv_dim, norm=opts.norm)
    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    # todo: B&W add your own initialization here
    G_XtoY.apply(weights_init)
    G_YtoX.apply(weights_init)
    D_X.apply(weights_init)
    D_Y.apply(weights_init)
    
    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_XtoY_path = os.path.join(opts.checkpoint_dir, 'G_XtoY_iter%d.pkl' % iteration)
    G_YtoX_path = os.path.join(opts.checkpoint_dir, 'G_YtoX_iter%d.pkl' % iteration)
    D_X_path = os.path.join(opts.checkpoint_dir, 'D_X_iter%d.pkl' % iteration)
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y_iter%d.pkl' % iteration)
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def merge_images(sources, targets, opts, k=10):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(opts.batch_size))
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    return merged.transpose(1, 2, 0)


def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)

    X, fake_X = utils.to_data(fixed_X), utils.to_data(fake_X)
    Y, fake_Y = utils.to_data(fixed_Y), utils.to_data(fake_Y)

    merged = merge_images(X, fake_Y, opts)
    logger.add_image("Fake_Y", img_tensor=(merged+1)/2, global_step=iteration, dataformats="HWC")
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X, opts)
    logger.add_image("Fake_X", img_tensor=(merged+1)/2, global_step=iteration, dataformats="HWC")
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))
    
def save_reconstructions(iteration, images_X, images_Y, reconstructed_X, reconstructed_Y, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    images_X, reconstructed_X = utils.to_data(images_X), utils.to_data(reconstructed_X)
    images_Y, reconstructed_Y = utils.to_data(images_Y), utils.to_data(reconstructed_Y)
    
    merged = merge_images(images_X, reconstructed_X, opts)
    path = os.path.join(opts.sample_dir, 'reconstruction-{:06d}-X.png'.format(iteration))
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))

    merged = merge_images(images_Y, reconstructed_Y, opts)
    path = os.path.join(opts.sample_dir, 'reconstruction-{:06d}-Y.png'.format(iteration))
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))


def training_loop(dataloader_X, dataloader_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)

    # print('\n=========== PARAMS =================')
    # for name, param in G_XtoY.named_parameters():
    #     print('name: ', name)
    #     print('param: ', param.shape)


    # Load generators, TODO: optimize this pipeline
#     G_XtoY.load_state_dict(torch.load('checkpoints_cyclegan/pokemon_water_normal/G_YtoX_iter20000.pkl'))
#     G_XtoY.load_state_dict(torch.load('checkpoints_cyclegan/pokemon_water_normal/G_XtoY_iter20000.pkl'))
    
#     D_X.load_state_dict(torch.load('checkpoints_cyclegan/pokemon_water_normal/D_X_iter20000.pkl'))
#     D_Y.load_state_dict(torch.load('checkpoints_cyclegan/pokemon_water_normal/D_Y_iter20000.pkl'))

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])


    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    fixed_X = utils.to_var(iter_X.next()[0])
    fixed_Y = utils.to_var(iter_Y.next()[0])

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    GX_losses = []
    GY_losses = []
    
    DX_reals = []
    DY_reals = []
    
    DX_fakes = []
    DY_fakes = []
    
    D_losses = []
    for iteration in range(1, opts.train_iters+1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, labels_X = iter_X.next()
        images_X, labels_X = utils.to_var(images_X), utils.to_var(labels_X).long().squeeze()

        images_Y, labels_Y = iter_Y.next()
        images_Y, labels_Y = utils.to_var(images_Y), utils.to_var(labels_Y).long().squeeze()
        
        # Differentiable Augmentation
        #images_X = DiffAugment(images_X, policy='translation,cutout')
        #images_Y = DiffAugment(images_Y, policy='translation,cutout')
        
        bs_x = images_X.shape[0]
        bs_y = images_Y.shape[0]

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        # Train with real images
        d_optimizer.zero_grad()

        # ==============================================================================================
        # ==================== 1. Compute the discriminator losses on real images ======================
        # ==============================================================================================
        if opts.label_smooth:
            label_real_x = torch.FloatTensor(bs_x).uniform_(0.8, 0.9).to(device)
            label_real_y = torch.FloatTensor(bs_y).uniform_(0.8, 0.9).to(device)

        else:
            label_real_x = torch.ones(bs_x, dtype=torch.float, device=device)
            label_real_y = torch.ones(bs_y, dtype=torch.float, device=device)

        label_real_x = torch.unsqueeze(label_real_x, dim=-1)
        label_real_x = torch.unsqueeze(label_real_x, dim=-1)

        label_real_y = torch.unsqueeze(label_real_y, dim=-1)
        label_real_y = torch.unsqueeze(label_real_y, dim=-1)

        # D_X_loss = torch.nn.functional.mse_loss(D_X(images_X), label_real_x, reduce='mean')# / bs_x
        D_X_loss = torch.mean(torch.square(D_X(images_X) - label_real_x))
        # D_Y_loss = torch.nn.functional.mse_loss(D_Y(images_Y), label_real_y, reduce='mean')# / bs_y
        D_Y_loss = torch.mean(torch.square(D_Y(images_Y) - label_real_y))

        d_real_loss = D_X_loss + D_Y_loss
        d_real_loss.backward()
        d_optimizer.step()
        
        # logger 
        logger.add_scalar('D/XY/real', D_X_loss, iteration)
        logger.add_scalar('D/YX/real', D_Y_loss, iteration)
        DX_reals.append(D_X_loss.item())
        DY_reals.append(D_Y_loss.item())

        d_optimizer.zero_grad()

        # ==============================================================================================
        # ==================== 2. Compute the discriminator losses on fake images ======================
        # ==============================================================================================

        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)
        #fake_X = DiffAugment(fake_X, policy='translation,cutout')
        label_fake_x = torch.zeros(bs_y, dtype=torch.float, device=device)
        label_fake_x = torch.unsqueeze(label_fake_x, dim=-1)
        label_fake_x = torch.unsqueeze(label_fake_x, dim=-1)
        
        # 3. Compute the loss for D_X
        # D_X_loss = torch.nn.functional.mse_loss(D_X(fake_X.detach()), label_fake_x, reduce='mean')# / bs_y
        D_X_loss = torch.mean(torch.square(D_X(fake_X) - label_fake_x))
        
        # 4. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)
        #fake_Y = DiffAugment(fake_Y, policy='translation,cutout')
        label_fake_y = torch.zeros(bs_x, dtype=torch.float, device=device)
        label_fake_y = torch.unsqueeze(label_fake_y, dim=-1)
        label_fake_y = torch.unsqueeze(label_fake_y, dim=-1)
        
        # 5. Compute the loss for D_Y
        # D_Y_loss = torch.nn.functional.mse_loss(D_Y(fake_Y.detach()), label_fake_y, reduce='mean')# / bs_x
        D_Y_loss = torch.mean(torch.square(D_Y(fake_Y) - label_fake_y))

        d_fake_loss = D_X_loss + D_Y_loss

        # Updating discriminator on fake every other iteration
        if iteration % 2 == 0:
            d_fake_loss.backward()
            d_optimizer.step()
            
        logger.add_scalar('D/XY/fake', D_X_loss, iteration)
        logger.add_scalar('D/YX/fake', D_Y_loss, iteration)
        DX_fakes.append(D_X_loss.item())
        DY_fakes.append(D_Y_loss.item())
    
        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        #########################################
        ########   Y--X-->Y CYCLE     ###########
        #########################################
        g_optimizer.zero_grad()

        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)
        print(fake_X.shape)
        #fake_X = DiffAugment(fake_X, policy='translation,cutout')

        if opts.label_smooth:
            label_fake_x = torch.FloatTensor(bs_y).uniform_(0.8, 0.9).to(device)
        else:
            label_fake_x = torch.ones(bs_y, dtype=torch.float, device=device)

        label_fake_x = torch.unsqueeze(label_fake_x, dim=-1)
        label_fake_x = torch.unsqueeze(label_fake_x, dim=-1)
        
        # 2. Compute the generator loss based on domain X
        # gY_loss = torch.nn.functional.mse_loss(D_X(fake_X), label_fake_x, reduce='mean')# / bs_y
        gY_loss = torch.mean(torch.square(D_X(fake_X) - label_fake_x))
        logger.add_scalar('G/XY/fake', gY_loss, iteration)

        if opts.use_cycle_consistency_loss:
            reconstructed_Y = G_XtoY(fake_X)
            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss_ = torch.nn.functional.l1_loss(images_Y, reconstructed_Y, reduce='mean') #/ bs_y
            gY_loss += opts.lambda_cycle * cycle_consistency_loss_
            logger.add_scalar('G/XY/cycle', opts.lambda_cycle * cycle_consistency_loss_, iteration)

#         g_loss.backward()
#         g_optimizer.step()
        GY_losses.append(gY_loss.item())

        #########################################
        ##    FILL THIS IN: X--Y-->X CYCLE     ##
        #########################################

        g_optimizer.zero_grad()

        # 1. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)
        #fake_Y = DiffAugment(fake_Y, policy='translation,cutout')
        if opts.label_smooth:
            label_fake_y = torch.FloatTensor(bs_x).uniform_(0.8, 0.9).to(device)
        else:
            label_fake_y = torch.ones(bs_x, dtype=torch.float, device=device)

        label_fake_y = torch.unsqueeze(label_fake_y, dim=-1)
        label_fake_y = torch.unsqueeze(label_fake_y, dim=-1)
        
        # 2. Compute the generator loss based on domain Y
        # gX_loss = torch.nn.functional.mse_loss(D_Y(fake_Y), label_fake_y, reduce='mean') #/ bs_x
        gX_loss = torch.mean(torch.square(D_Y(fake_Y) - label_fake_y))
        logger.add_scalar('G/YX/fake', gX_loss, iteration)

        if opts.use_cycle_consistency_loss:
            reconstructed_X = G_YtoX(fake_Y)
            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.nn.functional.l1_loss(images_X, reconstructed_X, reduce='mean')# / bs_x
            gX_loss += opts.lambda_cycle * cycle_consistency_loss
            logger.add_scalar('G/YX/cycle', cycle_consistency_loss, iteration)

        g_loss = gX_loss + gY_loss
        g_loss.backward()
        g_optimizer.step()
        GX_losses.append(gX_loss.item())


        # =========================================
        #            PLOTTING AND SAVING
        # =========================================

        # Print the log info and plot graphs 
        if iteration % opts.log_step == 0:
            print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
                  'd_fake_loss: {:6.4f} | g_loss: {:6.4f} | cycle_loss X->Y->X: {:6.4f} | cycle_loss Y->X->Y: {:6.4f}'.format(
                    iteration, opts.train_iters, d_real_loss.item(), D_Y_loss.item(),
                    D_X_loss.item(), d_fake_loss.item(), g_loss.item(), cycle_consistency_loss.item(), cycle_consistency_loss_.item()))

            # Plot training curves
            step_bins = 20
            num_examples = (len(GY_losses) // step_bins) * step_bins
                
            plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(GY_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="GYtoX Loss", 
                    color='slateblue'
                )
                
            plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(GX_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="GXtoY Loss", 
                    color='cornflowerblue'
                )
                 
            plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(DX_reals[:num_examples]).view(-1, step_bins).mean(1),
                    label="D_X Real",
                    color='mediumturquoise'
                )
            
            plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(DY_reals[:num_examples]).view(-1, step_bins).mean(1),
                    label="D_Y Real",
                    color='limegreen'
                )
                
            plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(DY_fakes[:num_examples]).view(-1, step_bins).mean(1),
                    label="D_Y Fake",
                    color='mediumvioletred'
                )
            
            plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(DX_fakes[:num_examples]).view(-1, step_bins).mean(1),
                    label="D_X Fake",
                    color='gold'
                )

            name = '{:s}-{:06d}.png'.format('Training Losses', iteration)
            path = os.path.join(opts.sample_dir, name)
            plt.ylim([0, 2])
            plt.legend(loc='upper right')
            plt.legend()
            plt.title('CycleGAN training loss')                
            plt.savefig(path, dpi=300)
            plt.show()
            plt.clf()
        
        
        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)
            save_reconstructions(iteration, images_X, images_Y, reconstructed_X, reconstructed_Y, opts)
        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create  dataloaders for images from the two domains X and Y
    dataloader_X = get_data_loader(opts.X, opts=opts)
    dataloader_Y = get_data_loader(opts.Y, opts=opts)
    print('dataloaders created')

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, dataloader_Y, opts)


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64, help='The side length N to convert images to NxN.')
    parser.add_argument('--disc', type=str, default='dc')
    parser.add_argument('--gen', type=str, default='cycle')
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--norm', type=str, default='instance')
    parser.add_argument('--use_cycle_consistency_loss', action='store_true', default=True, help='Choose whether to include the cycle consistency term in the loss.')
    parser.add_argument('--init_zero_weights', action='store_true', default=False, help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')
    parser.add_argument('--init_type', type=str, default='naive')
    parser.add_argument('--patch_disc', type=bool, default=True)
    parser.add_argument('--label_smooth', type=bool, default=True)

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=20000, help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_cycle', type=float, default=10)

    # Data sources
    parser.add_argument('--X', type=str, default='/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/spring2021'
                                                 '/16726_learning_based_image_synthesis/homeworks/hw3/16726_s21_hw3-main/data/cat/grumpifyAprocessed',
                        help='Choose the type of images for domain X.')
    parser.add_argument('--Y', type=str, default='/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/spring2021'
                                                 '/16726_learning_based_image_synthesis/homeworks/hw3/16726_s21_hw3-main/data/cat/grumpifyBprocessed',
                        help='Choose the type of images for domain Y.')
    parser.add_argument('--ext', type=str, default='*.png', help='Choose the type of images for domain Y.')
    parser.add_argument('--data_aug', type=str, default='deluxe', help='basic / none/ deluxe')

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_cyclegan/test_cats')
    parser.add_argument('--sample_dir', type=str, default='cyclegan/cats')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=100)
    parser.add_argument('--checkpoint_every', type=int , default=800)

    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--blocks', type=int, default=3)

    parser.add_argument('--patch', type=int, default=16)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()
    print(f"RUN: {vars(opts)}")

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    opts.sample_dir = os.path.join('output/', opts.sample_dir,
                                   '%s_%g' % (opts.X.split('/')[0], opts.lambda_cycle))
    opts.sample_dir += '%s_%s_%s_%s_%s' % (opts.data_aug, opts.norm, opts.disc, opts.gen, opts.init_type)
    if opts.use_cycle_consistency_loss:
        opts.sample_dir += '_cycle'

    opts.sample_dir += "_patch_{}_blocks_{}".format(opts.patch, opts.blocks)

    opts.patch_dim = (opts.image_size // opts.patch // 4) ** 2

    if os.path.exists(opts.sample_dir):
        cmd = 'rm %s/*' % opts.sample_dir
        os.system(cmd)


    logger = SummaryWriter(opts.sample_dir)

    print_opts(opts)
    main(opts)
