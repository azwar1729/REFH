
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
import model_2
from matplotlib import pyplot as plt
import generate_img
from generate_balls import generate
from neural_models.probabilistic_population_codes import LTIPPCs



def update_function(p, grad, loss, epoch,i):
  global mdot 
  epoch = epoch//400
  MaxEpoch = 500
  Ts = 0.01
  m0 = 20000
  b0 = 160000
  m = (m0*(Ts**2))/(1-(epoch-1)/MaxEpoch)
  b = (b0*(Ts**2))/(1-(epoch-1)/MaxEpoch)
  mm = m/Ts

  mdot[i] = (0.98-b/mm)*mdot[i] - grad/mm

  return p + Ts*mdot[i]

################### Hyperparameters #############################
repo = "results_RTRBM_Makin/"   #Path to folder where all results will be saved
epochs = 200000
T = 100
CD = 25  # No of CD steps 
MODEL = "TRBM" 
dataset = "BouncingBall"
z_dim = 400
y_dim = 900
OPTIM = "Makin"
lr = 5e-4  ## Use 1e-4 if dataset is BouncingBall and 1e-3 if dataset is PPC
BatchIsTrajectory = 1
num_trajectories = 1 # No of trajectories in training data, 
                     # If BatchIsTrajectory = 1, set num_trajectories = 1
BPTT = True  # If True, use BPTT, else without BPTT
             # If BPTT is True make sure BatchIsTrajectory = 1
#################################################################

def printf(*content,folder_path = repo + 'output.txt'):
    res = ''
    for i in content:

      res = res + ' ' +str(i)
    f= open(folder_path,"a")
    f.write(f'{res}\n')
    f.close()

################################################################
if dataset == "PPC":
  rbm = model_2.RBM(y_dim=15, u_dim=z_dim, z_dim=z_dim).cuda()
elif dataset == "BouncingBall":
  #rbm = model_2.RBM(y_dim=30*30, u_dim=z_dim, z_dim=z_dim).float().cuda()
  rbm = torch.load("rbm.pth")
  
if OPTIM == "Adam":
  optimizer = optim.Adam(rbm.parameters(),lr=lr)
  #optimizer = optim.SGD(rbm.parameters(),lr=1e-2,momentum=0.9)
elif OPTIM == "Makin":
  mdot = [torch.zeros_like(p) for p in rbm.parameters()]
  
if dataset == "PPC":
  PPCs = LTIPPCs()

error = [] 
print("Training Starting ....")
for epoch in tqdm(range(0,epochs)):
  
  if dataset == "BouncingBall":
    if epoch%5==0:
      train_data = generate(T = T, N = num_trajectories) 
      train_data = torch.from_numpy(train_data).float().cuda()
 
  
  elif dataset == "PPC":
    train_data = PPCs.get_data(num_trajectories=num_trajectories, T=1000)
    X = PPCs.latent_state
    train_data = torch.from_numpy(train_data).float().cuda()
  
  #print("epoch = ",epoch,"Done Generating Data ....")
  
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
          u_data_t = model_2.z_given_uy(u_data_t,train_data[:,t-1],rbm)
        elif BPTT == False:
          u_data_t = model_2.z_given_uy(u_data_t,train_data[:,t-1],rbm).detach().clone().cpu()
          u_data_t = u_data_t.cuda()
        u_data.append(u_data_t)
        
    else:
      u_data_t = torch.zeros(data_t.shape[0],z_dim).float().cuda()
      u_data.append(u_data_t)

    if BatchIsTrajectory == 0:
      
      y_0,u_0,z_0,y_1,u_1,z_1,suf_z = model_2.sample(data_t,u_data_t,rbm,dataset, k=CD, MODEL = MODEL)
      loss = model_2.Energy(u_data_t,data_t,suf_z,rbm) - model_2.Energy(u_data_t,y_1,z_1,rbm)
      loss = -loss.sum()
        
      if OPTIM == "Adam": 
        optimizer.zero_grad()
        loss.backward()
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
    #print(u_data.requires_grad)
    y_0,u_0,z_0,y_1,u_1,z_1,suf_z = model_2.sample(data,u_data,rbm,dataset, k=CD, MODEL = MODEL)
    grad_Wyz_2 = (torch.einsum("bi,bj->bij",[suf_z,data]).sum(axis=0) - torch.einsum("bi,bj->bij",[z_1,y_1]).sum(axis=0))
    grad_Wuz_2 = (torch.einsum("bi,bj->bij",[suf_z,u_data]).sum(axis=0) - torch.einsum("bi,bj->bij",[z_1,u_data]).sum(axis=0))
    loss = model_2.Energy(u_data,data,suf_z,rbm) - model_2.Energy(u_data,y_1,z_1,rbm)
    loss = -loss.sum()
    #print("epoch = ",epoch,"loss = ",loss,rbm.W_yz.mean().item(),rbm.W_uz.mean().item())
    #print("epoch = ",epoch,"grad = ",rbm.W_yz.grad.mean().item(),rbm.W_uz.grad.mean().item())
    #print("epoch = ",epoch,rbm.b_u.grad.mean().item(), rbm.b_y.grad.mean().item(), rbm.b_z.grad.mean().item())
   
    if OPTIM == "Adam": 
      optimizer.zero_grad()
      loss.backward()
      #torch.nn.utils.clip_grad_norm_(rbm.parameters(), max_norm=10)
      optimizer.step()
      
      #print("epoch = ",epoch,"loss = ",loss.item(),u_data.mean())
      #print("epoch = ",epoch,"grad = ",rbm.W_yz.grad.mean().item(),"grad_2 = ",grad_Wyz_2.mean().item())
      #print("epoch = ",epoch,"grad = ",rbm.W_uz.grad.mean().item(),"grad_2 = ",grad_Wuz_2.mean().item())
      #print("epoch = ",epoch, rbm.b_y.grad.mean().item(), rbm.b_z.grad.mean().item())
     
    elif OPTIM == "Makin":
      rbm.zero_grad()
      loss.backward()
      #print("epoch = ",epoch,"loss = ",loss.item(),u_data.mean())
      #print("epoch = ",epoch,"grad = ",rbm.W_yz.grad.mean().item(),rbm.W_uz.grad.mean().item())
      #print("epoch = ",epoch, rbm.b_y.grad.mean().item(), rbm.b_z.grad.mean().item())
      with torch.no_grad():
        for i, p in enumerate(rbm.parameters()):
          new_val = update_function(p, p.grad, loss, epoch,i)
          p.copy_(new_val)
    #print("epoch = ",epoch,"loss = ",loss.item(),rbm.W_yz.mean().item(),rbm.W_uz.mean().item())
    recon = model_2.y_given_z(suf_z,rbm).detach().cpu().numpy()
    
  if dataset == "BouncingBall":
    images = [(255*recon[i]).reshape(30,30).astype('uint8') for i in range(len(recon))]
    imgs = [Image.fromarray(img) for img in images]
    #imgs[0].save(f"results/array_recon{epoch//10}.gif", save_all=True, append_images=imgs[1:], loop=0)
    if epoch%1000==0:
      imgs[0].save(repo + f"array_recon{epoch}.gif", save_all=True, append_images=imgs[1:], loop=0)
      e1,e2 = model_2.predict(train_data,z_dim,rbm)
      print("epoch = ",epoch,"model loss = ",e1,"0th Order Model loss =",e2)
      #torch.save(rbm,"rbm.pth")
    
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
      print("PPC error = ",e1,e2)
    