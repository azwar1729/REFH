
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

folder_path = "/scratch/gilbreth/abdulsal/REFH/TRBM_2"
def printf(*content,folder_path = folder_path + '/' + 'output.txt'):

    res = ''
    for i in content:

      res = res + ' ' +str(i)
      
    f= open(folder_path,"a")
    f.write(f'{res}\n')
    f.close()
  


###Mini batch -100
## k =25
## T = 400
## generate new data 
## Max epoch = 255
# use means 
## add noisw 10% while generating 
# frame 50fps

z_dim = 625
epochs = 1000
T = 100
model = "REFH"
#rbm = torch.load('rbm.pth')
rbm = model_2.RBM(y_dim=32*32, u_dim=z_dim, z_dim=z_dim).cuda()
optimizer = optim.Adam(rbm.parameters(),lr=1e-4)

mdot = [torch.zeros_like(p) for p in rbm.parameters()]


for epoch in range(epochs):
  
  train_data = generate(T=T)
  print('DONE Generating New Data')
  train_data = torch.from_numpy(train_data).float().cuda()
  images = []
  recon_loss = 0
  loss_cum = torch.tensor(0)
  for t in range(T-1):

    data = train_data[:,t]

    #data = data.view(-1,1,32,32).float()

    if t>=1:
      u_data = suf_z.detach()
      #print("1 = ", u_data.requires_grad)
      #print("2 = ",suf_z.requires_grad)

    else:

      u_data = torch.zeros(data.shape[0],z_dim).cuda()
  
    y_0,u_0,z_0,y_1,u_1,z_1,suf_z = model_2.sample(data,u_data,rbm,k=25)

    recon_loss = recon_loss + F.mse_loss(y_0,y_1,reduction='sum')

    if model == "TRBM":
      loss = model_2.Energy(u_data,data,suf_z,rbm) - model_2.Energy(u_data,y_1,z_1,rbm)
    else:
      loss = model_2.Energy(u_data,data,suf_z,rbm) - model_2.Energy(u_1,y_1,z_1,rbm) 
    
    loss = -loss.mean()
    loss_cum = loss_cum + loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    ################################################################################################
    
    
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

    """
    rbm.zero_grad()
    loss.backward()
    with torch.no_grad():
      for i, p in enumerate(rbm.parameters()):
        new_val = update_function(p, p.grad, loss, epoch,i)
        p.copy_(new_val)
    
    """

    ################################################################################################

    
    #print(loss,f'epoch = {epoch}, T = {t}')
    #print('recon_loss = ', recon_loss)
    #print(log_var,mu)

    images.append((255*y_1[0]).cpu().detach().numpy().reshape(32,32).astype('uint8'))

  #optimizer.zero_grad()
  #loss_cum.backward()
  #optimizer.step()
  """
  rbm.zero_grad()
  loss_cum.backward()
  with torch.no_grad():
    for i, p in enumerate(rbm.parameters()):
      new_val = update_function(p, p.grad, loss, epoch,i)
      p.copy_(new_val)
  """  
  predicted = model_2.clamped_sampling(data,suf_z,rbm,k=100)
  actual = train_data[:,T-1]
  prior =  train_data[:,T-2]
  print('1 = ',((predicted - actual)**2).mean())
  print('2 = ',((prior - actual)**2).mean())
    
  if epoch%10==0:

      imgs = [Image.fromarray(img) for img in images]
      imgs[0].save(f"results/array_recon{epoch//1}.gif", save_all=True, append_images=imgs[1:], loop=0)

      #generate_img.generate(rbm,suf_z,epoch,"/scratch/gilbreth/abdulsal/REFH/TRBM_1")
      
      plt.imshow(predicted[0].detach().cpu().numpy().reshape(32,32))
      plt.savefig(f"results/predicted{epoch}")
      

torch.save(rbm,'rbm.pth')