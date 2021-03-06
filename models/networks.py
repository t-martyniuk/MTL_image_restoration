import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from models.fpn import FPNNet
from models.unet_seresnext import UNetSEResNext
from models.fpn_densenet import FPNDense
from models.fpn_inception import FPNEncoder, FPNDecoder
###############################################################################
# Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

class ResNetEncoder(nn.Module):
    def __init__(self, input_nc=3, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResNetEncoder, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output

class ResNetDecoder(nn.Module):
    def __init__(self, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d):
        super(ResNetDecoder, self).__init__()
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        n_downsampling = 2

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output

class EncoderDecoder:
    def __init__(self, encoder, decoder, learn_residual = False):
        self.learn_residual = learn_residual
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        enc = self.encoder(input)
        output = self.decoder(enc)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min = -1, max = 1)
        return output


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, use_parallel = True, learn_residual = False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output,min = -1,max = 1)
        return output



# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, gpu_ids=[], use_parallel = True):
        super(NLayerDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def get_nets(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'resnet':
        model_g = ResnetGenerator(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                  use_dropout=model_config['dropout'],
                                  n_blocks=model_config['blocks'],
                                  learn_residual=model_config['learn_residual'])
    elif generator_name == 'fpn':
        res = ResnetGenerator(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                  use_dropout=model_config['dropout'],
                                  n_blocks=6,
                                  learn_residual=False)
        model_g = FPNNet(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                         pretrained=model_config['pretrained'],
                         resnet=res)
    elif generator_name == 'fpn_inception':
        res = ResnetGenerator(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                              use_dropout=model_config['dropout'],
                              n_blocks=6,
                              learn_residual=False)
        model_g = FPNInception(res, norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    elif generator_name == 'fpn_dense':
        model_g = FPNDense()
    elif generator_name == 'unet_seresnext':
        model_g = UNetSEResNext(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                pretrained=model_config['pretrained'])
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)

    discriminator_name = model_config['d_name']
    if discriminator_name == 'n_layers':
        model_d = NLayerDiscriminator(n_layers=model_config['d_layers'],
                                      norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                      use_sigmoid=(model_config['disc_loss'] == 'gan'))
    else:
        raise ValueError("Discriminator Network [%s] not recognized." % discriminator_name)

    return nn.DataParallel(model_g), nn.DataParallel(model_d)


def get_nets_multitask(model_config, config):
    num_of_tasks = len(config['datasets'])
    discriminator_name = model_config['d_name']
    if discriminator_name == 'n_layers':
        discs = []
        for _ in range(num_of_tasks):
            discs.append(NLayerDiscriminator(n_layers=model_config['d_layers'],
                                       norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                       use_sigmoid=(model_config['disc_loss'] == 'gan')))
        discs = [nn.DataParallel(x) for x in discs]
        # model_d1 = NLayerDiscriminator(n_layers=model_config['d_layers'],
        #                                norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
        #                                use_sigmoid=(model_config['disc_loss'] == 'gan'))
        # model_d2 = NLayerDiscriminator(n_layers=model_config['d_layers'],
        #                                norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
        #                                use_sigmoid=(model_config['disc_loss'] == 'gan'))
    else:
        raise ValueError("Discriminator Network [%s] not recognized." % discriminator_name)

    generator_name = model_config['g_name']
    if generator_name == 'resnet':
        decs = []
        encoder = ResNetEncoder(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                use_dropout=model_config['dropout'],
                                n_blocks=model_config['blocks'])
        for _ in range(num_of_tasks):
            decs.append(ResNetDecoder(norm_layer=get_norm_layer(norm_type=model_config['norm_layer'])))
        decs = [nn.DataParallel(x) for x in decs]
        # decoder1 = ResNetDecoder(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
        # decoder2 = ResNetDecoder(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    elif generator_name == 'fpn_inception':
        decs = []
        encoder = FPNEncoder()
        for _ in range(num_of_tasks):
            decs.append(FPNDecoder(norm_layer=get_norm_layer(norm_type=model_config['norm_layer'])))
        decs = [nn.DataParallel(x) for x in decs]
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    # return {'encoder': nn.DataParallel(encoder), 'decoder1': nn.DataParallel(decoder1),
    #         'decoder2': nn.DataParallel(decoder2)}, \
    #        {'discr1': nn.DataParallel(model_d1), 'discr2': nn.DataParallel(model_d2)}
    return {'encoder': nn.DataParallel(encoder), 'decoders': decs}, discs




