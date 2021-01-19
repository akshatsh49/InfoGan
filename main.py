from auxiliary import *

num_epoch=51
coefficient=1
g_loss=[]
d_loss=[]
l1_loss=[]
l1_loss_vanilla=[]

d=discriminator()
d_vanilla=discriminator()
g=generator()
g_vanilla=generator()
d=d.to(device)
g=g.to(device)
d_vanilla=d_vanilla.to(device)
g_vanilla=g_vanilla.to(device)

od=optim.Adam(d.parameters(),lr=1e-4)
og=optim.Adam(g.parameters(),lr=1e-4)
od_vanilla=optim.Adam(d_vanilla.parameters(),lr=1e-4)
og_vanilla=optim.Adam(g_vanilla.parameters(),lr=1e-4)

ones=torch.ones(batch_size,device=device)
zeros=torch.zeros(batch_size,device=device)

for epoch in range(1,num_epoch+2):
  start=time.time()
  for img,l in train_loader:  
    img=img.to(device)
    noise,idx,con_noise=noise_sample()
    x_dash=g(noise)
    #discriminator training
    od.zero_grad()
    for p in g.parameters():
      p.requires_grad=False
    loss=bceloss(d(img.view(batch_size,1,28,28))[0].view(-1) , ones) + bceloss(d(x_dash)[0].view(-1) , zeros)
    loss.backward()
    d_loss.append(loss.item())
    torch.nn.utils.clip_grad_value_(d.parameters(),10)
    od.step()
    for p in g.parameters():
      p.requires_grad=True

    x_dash=g_vanilla(noise)
    od_vanilla.zero_grad()
    for p in g_vanilla.parameters():
      p.requires_grad=False
    loss=bceloss(d_vanilla(img.view(batch_size,1,28,28))[0].view(-1) , ones) + bceloss(d_vanilla(x_dash)[0].view(-1) , zeros)
    loss.backward()
    torch.nn.utils.clip_grad_value_(d_vanilla.parameters(),10)
    od_vanilla.step()
    for p in g_vanilla.parameters():
      p.requires_grad=True

    # generator training
    noise,idx,con_noise=noise_sample()
    x_dash=g(noise)
    og.zero_grad()
    score,mu,logsigma,logits=d(x_dash)
    l1=log_prob(con_noise,mu,logsigma) - celloss(logits , idx)
    l1_loss.append(l1.item())
    loss=bceloss(score.view(-1), ones) - coefficient*(l1)
    loss.backward()
    g_loss.append(loss.item())
    torch.nn.utils.clip_grad_value_(g.parameters(),10)
    og.step()

    og_vanilla.zero_grad()
    x_dash=g_vanilla(noise)
    og_vanilla.zero_grad()
    score,mu,logsigma,logits=d_vanilla(x_dash)
    l1=log_prob(con_noise,mu,logsigma) - celloss(logits , idx)
    l1_loss_vanilla.append(max(l1.item(),-15))    #values capped for visibility on graph
    loss=bceloss(score.view(-1), ones)
    loss.backward()
    torch.nn.utils.clip_grad_value_(g_vanilla.parameters(),10)
    og_vanilla.step()


  if(epoch%1==0):
    samples(g,epoch,100)
    track_training(g_loss,d_loss,l1_loss,l1_loss_vanilla)
    # print(g.tconv1.weight.norm(2)/g.tconv1.weight.grad.norm(2)) # test for vanishing and exploding gradients
    print('Done epoch {} in time {:0.2f} sec'.format(epoch,time.time()-start))
  
  if(epoch%5==1):
    path=saved_models+'\model_{}.pt'.format(epoch)
    torch.save({'epoch' : epoch , 'generator_state_dict' : g.state_dict() , 'discriminator_state_dict' : d.state_dict() , 'og_state_dict' : og.state_dict() , 'od_state_dict' : od.state_dict()}, path)

file=open(g_l_file,'wb')
pickle.dump(g_loss,file)
file.close()
file=open(d_l_file,'wb')
pickle.dump(d_loss,file)
file.close()