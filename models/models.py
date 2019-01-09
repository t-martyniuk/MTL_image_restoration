import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
from util.metrics import PSNR
from skimage.measure import compare_ssim as SSIM
from PIL import Image
import cv2

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

    def get_acc(self, output, target, full=False):
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        return psnr, ssim

    def get_loss(self, mean_loss, mean_psnr, mean_ssim):
        return '{:.3f}; psnr={}; ssim={}'.format(mean_loss, mean_psnr, mean_ssim)


    def visualize_data(self, writer, config, data, outputs, niter, degrad_type):

        # try:
        #     #images = vutils.make_grid([input_image, result_image, gt_image])
        #     #images = vutils.make_grid(input_image)
        #     print(data['A'].size())
        #     images = (vutils.make_grid(torch.squeeze(data['A']).permute(1, 2, 0)) + 1) / 2.0 * 255.0
        #     writer.add_image('Images', images, niter)
        # except Exception as e:
        #     print(e)
        #     print('I fucked up', niter, degrad_type)
        #     pass

        input_image = data['A'][0].cpu().float().numpy()
        input_image = (np.transpose(input_image, (1, 2, 0)) + 1) / 2.0 * 255.0

        gt_image = data['B'][0].cpu().float().numpy()
        # print(gt_image.shape)
        gt_image = (np.transpose(gt_image, (1, 2, 0)) + 1) / 2.0 * 255.0

        result_image = outputs[0].detach().cpu().float().numpy()
        result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0



        input_image = input_image.astype('uint8')
        gt_image = gt_image.astype('uint8')
        result_image = result_image.astype('uint8')
        result_image = np.hstack((input_image, result_image, gt_image))




        folder_name = 'train_images_'+str(degrad_type)+'_'+str(config['experiment_desc'])
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        cv2.imwrite(os.path.join(folder_name,
                                 str(int(niter)) + '.png'),
                    cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))


def get_model(model_config):
    return DeblurModel()

