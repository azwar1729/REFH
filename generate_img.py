from PIL import Image

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np 


from PIL import Image

import model_2
import random 


def binary_to_int(binary_digit_arr):
    return np.array([[int(''.join(map(str, x)), 2) for x in row] for row in binary_digit_arr])


def generate(net,z,epoch,path):

  images = []
  #z = torch.randn(32,256).cuda()
  for i in range(400):

  
    y = model_2.y_given_z(z,net)

    images.append((255*y[0]).cpu().detach().numpy().reshape(32,32).astype('uint8'))

    z = model_2.u_given_z(z,net)

    ############################################################################################
    """
    num = random.uniform(0,1)
    if num<0.1:

      z = model.z_to_u(z,net)[1]
    
    else:

      z = model.z_to_u(z,net)[0]
    """
    ############################################################################################

  imgs = [Image.fromarray(img) for img in images]

  images = np.array(images)
  #write_gif(images, 'rgbbgr.gif', fps=5)
  imgs[0].save(path + f"/arraybackward{epoch}.gif", save_all=True, append_images=imgs[1:],duration=1, loop=0)
  #imgs[0].save(f"arraybackward.gif", save_all=True, append_images=imgs[1:],duration=1, loop=0)
  #files.download(f'array{epoch}.gif')



def generate_2(net,z,epoch):

  recon = []
  for i in range(300):
    y = model_2.y_given_z(z,net)

    #images.append((255*y[0]).cpu().detach().numpy().reshape(32,32).astype('uint8'))
    recon.append((torch.bernoulli(y)).int().detach().cpu().numpy())

    z = model_2.u_given_z(z,net)

  recon = np.array(recon)

  print(binary_to_int(recon)[0][::-1])

