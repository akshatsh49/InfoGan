from auxiliary import *

latest=saved_models+ "/" +sorted(os.listdir(saved_models),key=len)[-1]
g_=generator()
g_=g_.to(device)
checkpoint=torch.load(latest)
g_.load_state_dict(checkpoint['generator_state_dict'])
samples(g_,51,num_samples=64,path=test_samples)

for i in range(5):
    space_interpolation(g_,steps=23,title=i,interpolation='spherical')

g_.eval()
img_grid=torch.randn(5,10,1,28,28,device=device)
with torch.no_grad():
    for j in range(5):
        noise_z=torch.randn(1,62,device=device)
        con_noise=torch.randn(1,2,device=device)
        for i in range(10):
            dis_noise=torch.zeros(1,10,device=device)
            dis_noise[0][i]=1.0
            img=g_(torch.cat([noise_z,dis_noise,con_noise],dim=1))
            img_grid[j][i]=img
show(torchvision.utils.make_grid(img_grid.view(-1,1,28,28),nrow=10), num_images=10*5,title='fi',path=fi_folder)
g_.train()


make_animation_interpolation(g_,steps=500,path=si_folder,interpolation='linear',name='1')

show_training_over_time()