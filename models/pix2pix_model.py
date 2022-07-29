import pdb
import torch
import torch.nn as nn
import numpy as np
from util.image_pool import ImagePool
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .perceptual_loss import PerceptualLoss
from .focal_frequency_loss import FocalFrequencyLoss
from .L1_Charbonnier_loss import L1_Charbonnier_loss


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1','D_real','D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc + opt.output_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionPerceptual = PerceptualLoss()
            # self.criterionFrequency = FocalFrequencyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_world_blur = input['C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def upsampling(sekf,img, x, y):
        func = nn.Upsample(size=[x, y], mode='bilinear', align_corners=True)
        return func(img)

    def generate_noise(self,size, channels=1, type='gaussian', scale=2, noise=None):
        if type == 'gaussian':
            noise = torch.randn(channels, size[0], round(size[1]/scale), round(size[2]/scale))
            noise = self.upsampling(noise, size[1], size[2])
        if type =='gaussian_mixture':
            noise1 = torch.randn(channels, size[0], size[1], size[2]) + 5
            noise2 = torch.randn(channels, size[0], size[1], size[2])
            noise = noise1 + noise2
        if type == 'uniform':
            noise = torch.randn(channels, size[0], size[1], size[2])
        return noise * 10.


    def concat_noise(self,img, *args):
        noise = self.generate_noise(*args)
        if isinstance(img, torch.Tensor):
            noise = noise.to(img.device)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        mixed_img = torch.cat((img, noise), 1)
        return mixed_img

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        n,c,h,w = self.real_A.shape
        self.real_A_noise = self.concat_noise(self.real_A,(c,h,w),n)
        # if True:
        #     pdb.set_trace()
        self.fake_B = self.netG(self.real_A_noise)  # G(A)

    # def backward_D(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     pred_fake = self.netD(fake_AB.detach())
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Real
    #     real_AB = torch.cat((self.real_A, self.real_B), 1)
    #     pred_real = self.netD(real_AB)
    #     self.loss_D_real = self.criterionGAN(pred_real, True)
    #     #C
    #     real_C = torch.cat((self.fake_B, self.real_world_blur), 1)
    #     pred_real_C = self.netD(real_C)
    #     self.loss_D_real_world_C = self.criterionGAN(pred_real_C, True)
    #     # combine loss and calculate gradients
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real_world_C + self.loss_D_real) * 0.5
    #     self.loss_D.backward(retain_graph=True)
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_real_world_C = self.backward_D_basic(self.netD,self.real_world_blur,fake_B)
        # self.loss_D_real_syth = self.backward_D_basic(self.netD,self.real_B,self.fake_B)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)* self.opt.lambda_L1
        # self.loss_G_perceptual = self.criterionPerceptual.forward(self.fake_B, self.real_B)
        # self.loss_G_frequency = self.criterionFrequency.forward(self.fake_B, self.real_B)*0.0
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN +self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
