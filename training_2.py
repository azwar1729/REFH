
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from PIL import Image
import model_2
from matplotlib import pyplot as plt
import generate_img
from generate_balls import generate
from neural_models.probabilistic_population_codes import LTIPPCs



def update_function(p, grad, loss, epoch,i):

  global mdot 
  MaxEpoch = 10000
  Ts = 0.01
  m0 = 2000
  b0 = 16000
  m = (m0*(Ts**2))/(1-(epoch-1)/MaxEpoch)
  b = (b0*(Ts**2))/(1-(epoch-1)/MaxEpoch)
  mm = m/Ts

  mdot[i] = (0.98-b/mm)*mdot[i] - grad/mm

  return p + Ts*mdot[i]

################### Hyperparameters ###############
epochs = 10000
T = 1000
CD = 25  # No of CD steps 
MODEL = "TRBM" 
dataset = "BouncingBall"
z_dim = 625
y_dim = 900
OPTIM = "Adam"
lr = 5e-4  ## Use 1e-4 if dataset is BouncingBall and 1e-3 if dataset is PPC
BatchIsTrajectory = 1
num_trajectories = 1 # No of trajectories in training data, 
                     # If BatchIsTrajectory = 1, set num_trajectories = 1
BPTT = False # If True, use BPTT, else without BPTT
             # If BPTT is True make sure BatchIsTrajectory = 1
###################################################

if dataset == "PPC":
  rbm = model_2.RBM(y_dim=15, u_dim=z_dim, z_dim=z_dim).cuda()
elif dataset == "BouncingBall":
  rbm = model_2.RBM(y_dim=30*30, u_dim=z_dim, z_dim=z_dim).cuda()
  
if OPTIM == "Adam":
  optimizer = optim.Adam(rbm.parameters(),lr=lr)
elif OPTIM == "Makin":
  mdot = [torch.zeros_like(p) for p in rbm.parameters()]
  
if dataset == "PPC":
  PPCs = LTIPPCs()

error = [] 
for epoch in range(epochs):
  
  if dataset == "BouncingBall":
    train_data = generate(T = T, N = num_trajectories) 
    train_data = torch.from_numpy(train_data).float().cuda()
 
  
  elif dataset == "PPC":
    train_data = PPCs.get_data(num_trajectories=num_trajectories, T=1000)
    X = PPCs.latent_state
    train_data = torch.from_numpy(train_data).float().cuda()
  
  print("epoch = ",epoch,"Done Generating Data ....")
  
  u_data = []
  recon = []
  for t in range(T):

    data_t = train_data[:,t]  # data_t is a batch of frames for a given timestep
    if t>=1:
      
      if BatchIsTrajectory == 0:
        if BPTT == True:
          u_data_t = suf_z  # Set the sufficient stats from the previous time step as u_data
                            # Since suf_z is calculated before the gradient update it is in line with Makin's code
        elif BPTT == False:
          u_data_t = suf_z.detach().cpu().cuda() 
          
      if BatchIsTrajectory == 1:     
        if BPTT == True:
          u_data_t = model_2.z_given_uy(u_data[-1],train_data[:,t-1],rbm)
        elif BPTT == False:
          u_data_t = model_2.z_given_uy(u_data[-1],train_data[:,t-1],rbm).detach()
        u_data.append(u_data_t)
        
    else:
      u_data_t = torch.zeros(data_t.shape[0],z_dim).cuda()
      u_data.append(u_data_t)

    if BatchIsTrajectory == 0:
      
      y_0,u_0,z_0,y_1,u_1,z_1,suf_z = model_2.sample(data_t,u_data_t,rbm,dataset, k=CD, MODEL = MODEL)
      loss = model_2.Energy(u_data_t,data_t,suf_z,rbm) - model_2.Energy(u_data_t,y_1,z_1,rbm)
      loss = -loss.sum()
      optimizer.zero_grad()
      loss.backward()
      
      if OPTIM == "Adam": 
        optimizer.step()
        
      elif OPTIM == "Makin":
        
        rbm.zero_grad()
        loss.backward()
        with torch.no_grad():
          for i, p in enumerate(rbm.parameters()):
            new_val = update_function(p, p.grad, loss, epoch,i)
            p.copy_(new_val)
      
      y_mean = model_2.y_given_z(suf_z,rbm)
      recon.append((y_mean[0]).cpu().detach().numpy())
      
  if BatchIsTrajectory == 1:
    
    u_data = torch.stack(u_data)
    u_data = u_data.view(-1,z_dim)
    data = train_data.view(-1,y_dim)
   
    y_0,u_0,z_0,y_1,u_1,z_1,suf_z = model_2.sample(data,u_data,rbm,dataset, k=CD, MODEL = MODEL)
    loss = model_2.Energy(u_data,data,suf_z,rbm) - model_2.Energy(u_data,y_1,z_1,rbm)
    loss = -loss.sum()
    print("epoch = ",epoch,"loss = ",loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    recon = model_2.y_given_z(suf_z,rbm).detach().cpu().numpy()
    
  if dataset == "BouncingBall":
    images = [(255*recon[i]).reshape(30,30).astype('uint8') for i in range(len(recon))]
    imgs = [Image.fromarray(img) for img in images]
    print(recon.min(),recon.max())
    #imgs[0].save(f"results/array_recon{epoch//10}.gif", save_all=True, append_images=imgs[1:], loop=0)
    if epoch%1000==0:
      imgs[0].save(f"results/array_recon{epoch}.gif", save_all=True, append_images=imgs[1:], loop=0)
      model_2.predict(train_data,z_dim,rbm)
    
  if dataset == "PPC":
    recon = np.array(recon)
    xhat = PPCs.position_PPC.samples_to_estimates(recon)
    xhat_data = PPCs.position_PPC.samples_to_estimates(train_data[0,:,:].detach().cpu().numpy())
    x = X@PPCs.C.T
    x = x[0]
    e1 = ((xhat_data - x)**2).mean().item()
    e2 = ((xhat - x)**2).mean().item()
    error.append(e2)
    if epoch%10==0 :
      print(e1,e2)
    