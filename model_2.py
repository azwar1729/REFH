

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.transforms as T
import model_2
import matplotlib.pyplot as plt
import numpy as np
#0.005

class RBM(nn.Module):
   def __init__(self,y_dim=1024, u_dim=2000, z_dim=2000):
              
        super(RBM, self).__init__()
  
        self.W_yz = nn.Parameter(torch.randn(z_dim,y_dim)*1e-2)
        self.W_uz = nn.Parameter(torch.randn(z_dim,u_dim)*1e-2)

        self.b_y = nn.Parameter(torch.zeros(y_dim))
        self.b_u = nn.Parameter(torch.zeros(u_dim))
        self.b_z = nn.Parameter(torch.zeros(z_dim))



def y_given_z(z,rbm,dataset="BouncingBall"):
    temp = rbm.b_y + torch.einsum("ij,bj->bi",[rbm.W_yz.T,z])
    if dataset == "BouncingBall":
      p_y = 1/(1+torch.exp(-temp))
    elif dataset == "PPC":
      p_y = torch.exp(temp)
    return p_y


def u_given_z(z,rbm):
    temp = rbm.b_u + torch.einsum("ij,bj->bi",[rbm.W_uz.T,z])
    p_u = 1/(1+torch.exp(-temp))
    return p_u


def z_given_uy(u,y,rbm):
    temp = torch.einsum("ij,bj->bi",[rbm.W_yz,y]) + torch.einsum("ij,bj->bi",[rbm.W_uz,u]) + rbm.b_z
    p_z = 1/(1+torch.exp(-temp))
    return p_z

@torch.no_grad()
def sample(y,u,rbm,dataset,k=1,MODEL = "TRBM"):
  
  ## p_y, p_z, p_u represents means of probability distribution
  ## y, z, u represents samples
  ## y_0, u_0 are the intial data and p_z_0 is the mean of the hidden activity corresponding to y_0 and u_0
   
  y_0 = y
  u_0 = u
  p_z_0 = z_given_uy(u,y,rbm) ## get means of the hidden units
  z = torch.bernoulli(p_z_0)  ## get samples 
  z_0 = z.clone()
  
  for _ in range(k):

    ####### Sample Y_t ###############
    p_y = y_given_z(z,rbm,dataset=dataset)
    if dataset == "BouncingBall":
      y = torch.bernoulli(p_y)
    elif dataset == "PPC":
      y = torch.poisson(p_y)
    
    ####### Sample U_{t-1} ############
    if MODEL == "REFH":        # Sample previous sufficient statistic only if REFH
      p_u = u_given_z(z,rbm)
      u = torch.bernoulli(p_u)
    
    ####### Sample Z_t ################
    p_z = z_given_uy(u,y,rbm)
    z = torch.bernoulli(p_z)

  return y_0,u_0,z_0,y,u,p_z,p_z_0

def clamped_sampling(y,u,rbm,k=1):
  
  p_z_0 = z_given_uy(u,y,rbm)
  z = torch.bernoulli(p_z_0)
  diff  = 0 
  for _ in range(k):

    p_y = y_given_z(z,rbm)
    y_1 = torch.bernoulli(p_y)

    diff = ((y - y_1)).sum(axis=1).mean()
    #print("diff = ",diff)
    y = y_1
    p_z = z_given_uy(u,y,rbm)
    z = torch.bernoulli(p_z)
   
  return p_y

def predict(train_data,z_dim,rbm):

  T = len(train_data[0])
  loss_list = []
  loss_0thOrder_list = []
  for t in range(T-1):
    data_t = train_data[:,t]
    if t>=1:  
      u_data_t = suf_z  
    else:
      u_data_t = torch.zeros(data_t.shape[0],z_dim).cuda()
    
    p_y = clamped_sampling(data_t,u_data_t,rbm,k=25)
    loss = ((p_y - train_data[:,t+1])**2).mean().item()
    loss_0thOrder = ((train_data[:,t] - train_data[:,t+1])**2).mean().item()
    loss_list.append(loss)
    loss_0thOrder_list.append(loss_0thOrder)
    suf_z = z_given_uy(u_data_t,data_t,rbm)
  
  loss_list = np.array(loss_list)
  loss_0thOrder_list = np.array(loss_0thOrder_list)
  print("model loss = ",loss_list.mean(),"0th Order Model loss =",loss_0thOrder_list.mean())
  return loss_list.mean()
    
    
    
def Energy(u,y,z,rbm):

    energy = torch.einsum("ij,bj->bi",[rbm.W_yz,y]) + torch.einsum("ij,bj->bi",[rbm.W_uz,u]) + rbm.b_z
    energy = (z * energy).sum(axis=1) + (rbm.b_y * y).sum(axis=1) + (rbm.b_u * u).sum(axis=1)

    return energy


#rbm = RBM().cuda()
#print(sample(torch.randn(4,1024).cuda(),torch.randn(4,2000).cuda(),rbm))
#print(Energy(torch.randn(4,2000).cuda(),torch.randn(4,1024).cuda(),torch.randn(4,2000).cuda(),rbm))