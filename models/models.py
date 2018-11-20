import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
from util.metrics import PSNR, SSIM
import pytorch_ssim
from PIL import Image

class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['A']
        inputs = img
        targets = data['B']
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_acc(self, output, target):

        psnr = PSNR(self.tensor2im(output.data), self.tensor2im(target.data))

        return psnr

    def get_loss(self, mean_loss, mean_psnr, output=None, target=None):
        return '{:.3f}; psnr={}'.format(mean_loss, mean_psnr)

    def visualize_data(self, writer, data, outputs, niter):
        images = vutils.make_grid(data['A']) + 1 / 2.0
        writer.add_image('Images', images, niter)


def get_model(model_config):
    return DeblurModel()

