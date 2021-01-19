from model import *

celloss=nn.CrossEntropyLoss(reduction='mean')
bceloss=nn.BCELoss()

def track_training(g_loss,d_loss,l1_loss,l1_loss_vanilla):
  plt.close()
  plt.plot(range(1,1+len(g_loss)) , g_loss , label='Generator Loss')
  plt.legend()
  plt.title("Generator_Loss")
  plt.xlabel('Iterations')
  plt.savefig('track_loss\Generator_Loss.png')

  plt.close()
  plt.plot(range(1,1+len(d_loss)) , d_loss , label='Discriminator Loss')
  plt.legend()
  plt.title("Discriminator_Loss")
  plt.xlabel('Iterations')
  plt.savefig('track_loss\Discriminator_Loss.png')

  plt.close()
  plt.plot(range(1,1+len(l1_loss)) , l1_loss , label='L1_loss')
  plt.plot(range(1,1+len(l1_loss_vanilla)) , l1_loss_vanilla , label='L1_loss_vanilla_generator')
  plt.plot(range(1,1+len(l1_loss)) , len(l1_loss)*[0] ,label='Maximum_value')
  plt.legend()
  plt.title("L1_metric")
  plt.xlabel('Iterations')
  plt.yticks(np.arange(-12,1,step=1))
  plt.savefig('track_loss\L1_Loss.png')
  plt.close()

def noise_sample(size=batch_size,z_dim=62,no_c_dis=1,c_dis_dim=10,no_c_con=2):
  noise_z=torch.randn(size,z_dim,device=device)
  if(no_c_dis>0):
    dis_c=torch.zeros(size,no_c_dis,c_dis_dim,device=device)
    idx=np.random.randint(c_dis_dim,size=(size,no_c_dis))
    for i in range(size):
      for j in range(no_c_dis):
        dis_c[i][j][idx[i][j]]=1
  
    dis_c=dis_c.view(size,-1)
    noise_z=torch.cat([noise_z,dis_c],dim=1)
    idx_tensor=torch.from_numpy(idx).view(size,no_c_dis).long().view(1,-1)[0]
    idx_tensor=idx_tensor.to(device)

  if(no_c_con>0):
    con_noise=torch.randn(size,no_c_con,device=device)
    noise_z=torch.cat([noise_z,con_noise],dim=1)

  return noise_z,idx_tensor,con_noise

def show(img_tensor_grid,num_images=batch_size,title='',path=sample_folder):
  plt.close()
  npimg=img_tensor_grid.cpu().detach().numpy()
  plt.imshow(np.transpose(npimg,(1,2,0)) , interpolation='nearest')
  plt.title(title)
  plt.axis('off')
  plt.savefig(path+'\{}.png'.format(title))

def samples(g,epoch,num_samples=25,path=sample_folder):
  g.eval()
  noise,idx,cn=noise_sample(size=num_samples)
  xdash=g(noise)
  show(torchvision.utils.make_grid(xdash.view(num_samples,1,28,28),nrow=int(math.sqrt(num_samples))),num_images=num_samples,title='Epoch_{}'.format(epoch),path=path)
  g.train()

def log_prob(c,mu,logsigma):
  ans=-1*(mu.shape[1])/2*torch.log(torch.pi) - logsigma - 0.5*(c-mu).pow(2)/(torch.exp(logsigma).pow(2))
  return ans.mean(dim=0).sum()

def space_interpolation(g,steps=23,title='',interpolation='linear'):
  g.eval()
  with torch.no_grad():
    noise1,_,_=noise_sample(size=1)
    noise2,_,_=noise_sample(size=1)
    noise =[]
    if(interpolation=='linear'):
      for i in range(steps+1):
        noise.append(noise1+i/steps*(noise2-noise1))
      tensor_noise=torch.stack(noise,dim=0)
      show(torchvision.utils.make_grid(g(tensor_noise).view(-1,1,28,28),nrow=8), num_images=steps+1,title=title,path=si_folder)

    elif(interpolation=='spherical'):
      theta=torch.acos(torch.mul(noise1,noise2).view(-1).sum()/(noise1.norm(2)*noise2.norm(2)))
      for i in range(steps+1):
        t=i/steps
        noise.append(noise1*(torch.sin(t*theta)/torch.sin(theta))+noise2*(torch.sin((1-t)*theta)/torch.sin(theta)) )
      tensor_noise=torch.stack(noise,dim=0)
      show(torchvision.utils.make_grid(g(tensor_noise).view(-1,1,28,28),nrow=8), num_images=steps+1,title=title,path=si_folder)
  g.train()

def factor_interpolation(g,title=''):
  g.eval()
  with torch.no_grad():
    noise_z=torch.randn(1,62,device=device)
    con_noise=torch.randn(1,2,device=device)
    img_list=[]
    for i in range(10):
      dis_noise=torch.zeros(1,10,device=device)
      dis_noise[0][i]=1.0
      img=g(torch.cat([noise_z,dis_noise,con_noise],dim=1))
      img_list.append(img.view(1,28,28))
  show(torchvision.utils.make_grid(torch.stack([*img_list]),nrow=8), num_images=10,title=title,path=fi_folder)
  g.train()

def make_animation_interpolation(g,steps,path=si_folder,interpolation='linear',name=''):
  g.eval()
  fig=plt.figure()
  img_list=[]
  with torch.no_grad():
    noise1,_,_=noise_sample(size=1)
    noise2,_,_=noise_sample(size=1)
    noise =[]
    if(interpolation=='linear'):
      for i in range(steps+1):
        noise.append(noise1+(1-np.abs(1-2*i/steps))*(noise2-noise1))
        img_list.append([ plt.imshow(g(noise[-1]).view(28,28).cpu().detach().numpy(),animated=True)] )

    elif(interpolation=='spherical'):
      theta=torch.acos(torch.mul(noise1,noise2).view(-1).sum()/(noise1.norm(2)*noise2.norm(2)))
      for i in range(steps+1):
        t=i/steps
        noise.append(noise1*(torch.sin(t*theta)/torch.sin(theta))+noise2*(torch.sin((1-t)*theta)/torch.sin(theta)) )
        img_list.append([ plt.imshow(g(noise[-1]).view(28,28).cpu().detach().numpy(),animated=True)] )

  ani=matplotlib.animation.ArtistAnimation(fig,img_list,interval=1,blit=True,repeat_delay=0)
  ani.save(path+'/ani_'+ name+'.gif')
  plt.show()
  plt.close()
  g.train()
  
def show_training_over_time():
  img_list=[]
  g_=generator()
  g_=g_.to(device)
  fig=plt.figure()
  noise,_,_=noise_sample(size=1)
  for path in os.listdir(saved_models):
    path=saved_models+ "/" + path 
    checkpoint=torch.load(path)
    g_.load_state_dict(checkpoint['generator_state_dict'])
    g_.eval()
    img_list.append( [plt.imshow(g_(noise).view(28,28).cpu().detach().numpy() , animated=True)] )
  
  ani=matplotlib.animation.ArtistAnimation(fig,img_list,interval=500,blit=True,repeat_delay=1000)
  ani.save(test_samples +'/training_over_epochs.gif')
  plt.show()
  plt.close()
