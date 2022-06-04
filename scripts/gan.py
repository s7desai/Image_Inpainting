import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import os.path as osp
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data as D
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import os.path as osp
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data as D
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import cv2
from tqdm import tqdm
import os
import zipfile 
import torch
import time
import torchvision.transforms as T
import random as rnd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid,save_image
from torchvision.datasets import ImageFolder,DatasetFolder
from torch.utils.data import Dataset,DataLoader,Subset
from torch.autograd import Variable
from torchvision.transforms import InterpolationMode
from numpy.random import choice
from numpy.random import seed as np_seed


import re
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

bottleneck = 4000
image_size = 256

class Data(D.Dataset):
    def __init__(self, path_mask, path_gt):
        super(Data, self).__init__()
        self.masked = []
        self.gt = []
        self.path_mask = path_mask
        self.path_gt = path_gt
        self.transform = transforms.Compose([transforms.ToTensor()])
        files_mask = sorted(glob.glob(os.path.join(path_mask, '*.jpg')), key=numericalSort)[:5000]
        files_gt = sorted(glob.glob(os.path.join(path_gt, '*.jpg')), key = numericalSort)[:5000]
        for mask, gt in  zip(files_mask, files_gt):
            self.masked.append(mask)
            self.gt.append(gt)
        self.len = len(self.masked)
        
    def __getitem__(self, index):
        img_masked = Image.open(self.masked[index])
        img_gt = Image.open(self.gt[index])
        return self.transform(img_masked), self.transform(img_gt)
    
    def __len__(self):
        return self.len
    
path_mask = '/home/sghosal/Project/Image_Inpainting/data_places_mask/'
path_gt = '/home/sghosal/Project/Image_Inpainting/data_places_gt/'
imgs = Data(path_mask, path_gt)

train_loader = D.DataLoader(imgs, batch_size = 32, shuffle = True, num_workers = 0)

class Generator(nn.Module):
    def __init__(self,bottleneck):
        super(Generator, self).__init__()

        # Encoder

        # input: image_size * image_size * 3; image_size = 256
        self.encoder_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU(0.2,True)
        )
        # input: 128 * 128 * 64
        self.encoder_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        # input: 64 * 64 * 64
        self.encoder_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        # input: 32 * 32 * 64
        self.encoder_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True)
        )
        # input: 16 * 16 * 256
        self.encoder_layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True)
        )
        # input: 8 * 8 * 256
        self.encoder_layer_6 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,True)
        )
        # input: 4 * 4 * 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512,bottleneck,kernel_size=(4,4)), 
            nn.BatchNorm2d(bottleneck),
            nn.ReLU()
        )
        # current state: 1 * 1 * bottleneck
        
        # Decoder
        self.decoder_layer_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=bottleneck,out_channels=512,kernel_size=(4,4)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # Input: 4 * 4 * 512
        self.decoder_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Input: 8 * 8 * 256
        self.decoder_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # Input: 16 * 16 * 128
        self.decoder_layer_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # Input: 32 * 32 * 64
        self.decoder_layer_5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(4,4),stride=2,padding=1),
            nn.Tanh()
        )
        # Output: 64 * 64 * 3

    def forward(self,x):
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)
        x = self.encoder_layer_4(x)
        x = self.encoder_layer_5(x)
        x = self.encoder_layer_6(x)
        x = self.bottleneck(x)
        x = self.decoder_layer_1(x)
        x = self.decoder_layer_2(x)
        x = self.decoder_layer_3(x)
        x = self.decoder_layer_4(x)
        x = self.decoder_layer_5(x)

        return x
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Input: 64 * 64 * 3
        self.disc_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU(0.2,True)
        )
        # Input: 32 * 32 * 64
        self.disc_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        # Input: 16 * 16 * 128
        self.disc_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True)
        )
        # Input: 8 * 8 * 256
        self.disc_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,True)
        )
        # Input: 4 * 4 * 512
        self.disc_layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1,kernel_size=(4,4),stride=1,padding=0),
            nn.Sigmoid()
        )
        # Output: 1 * 1 * 1
    
    def forward(self,x):
        x = self.disc_layer_1(x)
        x = self.disc_layer_2(x)
        x = self.disc_layer_3(x)
        x = self.disc_layer_4(x)
        x = self.disc_layer_5(x)
        x = x.view(-1)
        return x

    
if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    Tensor = torch.FloatTensor
    device = 'cpu'

dev = device

# Number of channel RGB 3
channel = 3 

# Imgage size will be C*256*256
img_size = 256

# Mask size it's the size of center mask Cx64x64
mask_size = 64

# Number of pixels overlapped
overlapPred = 0

# Size of batches
batch_size = 32

#  the lower is res value, the more continuous the output will be.
## Value to generate a random patter of 1 and 0 to create a random region
res = 0.06
density = 0.25
MAX_SIZE = 10000

# Paths 
save_path_discriminator = "./gan/checkpoint_discriminator.pth"
save_path_generator = "./gan/checkpoint_generator.pth"

# Restore backups
restore = False


def weights_init_normal(m):
    # Initialize model
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
        
def create_model(bottleneck=4000, gen_lr=0.0002, dis_lr=0.0002, gen_weight_decay=5e-4, dis_weight_decay=5e-4):
    
    latent_dim = bottleneck
    g_net = Generator(latent_dim)
    d_net = Discriminator()
    g_optimizer = optim.Adam(g_net.parameters(), lr=gen_lr, betas=(0.5,0.999), weight_decay=gen_weight_decay)
    d_optimizer = optim.Adam(d_net.parameters(), lr=dis_lr, betas=(0.5,0.999), weight_decay=dis_weight_decay)
    
    g_net = g_net.to(dev)
    d_net = d_net.to(dev)
    return g_net,d_net,g_optimizer,d_optimizer




def training(train_loader,test_loader= None, labels_noise=False,wtl2= 0.999,last_epoch=200,save_photos_interval=10,overlapL2Weight=10, restore = False):
    '''
    train_loader: Dataloader of train data
    test_loader : Dataloader of test data
    labels_noise: Boolean that enable labels smoothing and flipping
    wtl2: param to weights losses 
    last_epoch: number of last epoch
    save_photos_interval: set interval of every x epoch generate photos to compare
    overlapL2Weight: weights amplified 

    '''
    # Define labels 
    
    path_toSave_photos = "/content/images/"
    total_time = 0
    # load test image
    # test_image= next(iter(test_loader))
    # test_image = test_image.to(dev)
    # test_masked_imgs =test_image.clone() 
    # Create the models 
    g_net,d_net,g_optimizer,d_optimizer = create_model()
    # If backup it's available load it 
    if os.path.isfile(save_path_discriminator) and os.path.isfile(save_path_generator) and restore:
        checkpoint_d =  torch.load(save_path_discriminator)
        checkpoint_g =  torch.load(save_path_generator)
        d_net.load_state_dict(checkpoint_d['d_state_dict'])
        d_optimizer.load_state_dict(checkpoint_d['optimizer_state_dict'])
        d_loss = checkpoint_d['loss']
        d_loss_fake = checkpoint_d['loss_fake']
        d_loss_real = checkpoint_d['loss_real']
        g_net.load_state_dict(checkpoint_g['g_state_dict'])
        g_optimizer.load_state_dict(checkpoint_g['optimizer_state_dict'])
        g_loss = checkpoint_g['loss']    
        g_loss_pixel = checkpoint_g['loss_pixel']
        g_loss_adv = checkpoint_g['loss_adv']
        epoch_backup = checkpoint_g['epoch']+1
        print("Discriminator and Generator restored")
    else:
        epoch_backup = 0 
        g_net.apply(weights_init_normal)
        d_net.apply(weights_init_normal)
        print("weight applied")
    try:
        for epoch in range(epoch_backup,last_epoch):
            # Losses
            start = time.time()
            sum_d_loss = 0
            sum_d_fake_loss = 0
            sum_d_real_loss = 0
            sum_g_loss = 0
            sum_g_loss_adv = 0
            sum_g_loss_pixel = 0
            # Training mode
            d_net.train()
            g_net.train()
            # Process all training batches
            i = 0
    
            for i,(x,y) in tqdm(enumerate(train_loader)):
                batch_length = x.shape[0]
#                 print(f'batch size : {batch_length}')
                valid = Variable(Tensor(batch_length).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(batch_length).fill_(0.0), requires_grad=False)
                x = x.to(device)
                y = y.to(device)
                # Move to device
                i+=1
#                 masked_parts = get_center(batch)
                masked_parts = y[:,:,96:160,96:160]
                #masked_parts are the center of the images 
                masked_parts = Variable(masked_parts.type(Tensor))
                masked_imgs = x.clone()
#                 masked_imgs = apply_center_mask(img_mask)
#                 masked_imgs = Variable(masked_imgs.type(Tensor))

                ### Discriminator 
                
                # Reset discriminator gradient
                d_optimizer.zero_grad()

                # Forward (discriminator, real)
                output = d_net(masked_parts) 
                # Compute loss (discriminator, real)
               
                d_real_loss =  F.binary_cross_entropy(output, valid)
                # Backward (discriminator, real)
                d_real_loss.backward()
                sum_d_real_loss += d_real_loss.item()  
                #generate sample from masked images         
                g_output = g_net(masked_imgs)
                # Forward (discriminator, fake; also generator forward pass)
                output = d_net(g_output.detach()) # This prevents backpropagation from going inside the generator
                # Compute loss (discriminator, fake)
                d_fake_loss = F.binary_cross_entropy(output, fake)
                # Backward (discriminator, fake)
                d_fake_loss.backward()
                sum_d_fake_loss += d_fake_loss.item()           
                d_loss = 0.5*(d_fake_loss + d_real_loss)
                sum_d_loss += d_loss.item()
                # Update discriminator
                d_optimizer.step()
                
                ### Generator 
                g_optimizer.zero_grad()
                # Forward (generator)
                output =  d_net(g_output)
                # Compute adversarial loss
                g_loss_adv = F.binary_cross_entropy(output, valid)            
                # Comput pixelwise loss
                # but amplifying weights 10x 
                #g_loss_pixel =  criterionMSE(g_output,masked_parts)
#                 wtl2Matrix = masked_parts.clone()
                # OverlapL2weight = 10
#                 wtl2Matrix.data.fill_(wtl2*overlapL2Weight)
#                wtl2Matrix.data[:,:,overlapPred:mask_size-overlapPred,overlapPred:mask_size-overlapPred] = wtl2
                # MSE Loss
                g_loss_pixel = (g_output-masked_parts).pow(2)
                # Multiply 
#                 g_loss_pixel = g_loss_pixel * wtl2Matrix
                g_loss_pixel = g_loss_pixel.mean()
#                 print(f'generator loss pixel : {g_loss_pixel}')
                # The losse it's the sum of adv and pixel
                g_loss = (1-wtl2) * g_loss_adv + wtl2 * g_loss_pixel
                sum_g_loss_adv += g_loss_adv.item()
                sum_g_loss_pixel += g_loss_pixel.item()
                sum_g_loss += g_loss.item()
                # Backward (generator)
                g_loss.backward()
                # Update generator
                g_optimizer.step()
                if epoch % 2 == 0:
                    torch.save(g_net.state_dict(), './gan/generator.pt')
                    torch.save({'g_state_dict': g_net.state_dict(),
                            'optimizer_state_dict': g_optimizer.state_dict(),
                            'loss': g_loss,
                            'loss_pixel': g_loss_pixel,
                            'loss_adv': g_loss_adv,
                            'epoch':epoch,
                            }, save_path_generator)

                    torch.save({'d_state_dict': d_net.state_dict(),
                                'optimizer_state_dict': d_optimizer.state_dict(),
                                'loss_fake': d_fake_loss,
                                'loss_real': d_real_loss,
                                'loss': d_loss,
                                'epoch': epoch
                                }, save_path_discriminator)
#                 if (i%700==0):
#                     print(f"Batches {i}/{len(train_loader)}")

            # Epoch end, print losses
            epoch_d_loss = sum_d_loss/len(train_loader)
            epoch_d_real_loss = sum_d_real_loss/len(train_loader)
            epoch_d_fake_loss = sum_d_fake_loss/len(train_loader)
            epoch_g_loss_adv = sum_g_loss_adv/len(train_loader)
            epoch_g_loss_pixel = sum_g_loss_pixel/len(train_loader)
            epoch_g_loss = sum_g_loss/len(train_loader)
            end = time.time()  
            print(f'generator loss : {epoch_g_loss} discriminator loss : {epoch_d_loss}')
            time_epoch = (end - start)/60
            total_time +=time_epoch
            # Save models
#             torch.save({'g_state_dict': g_net.state_dict(),
#                         'optimizer_state_dict': g_optimizer.state_dict(),
#                         'loss': g_loss,
#                         'loss_pixel': g_loss_pixel,
#                         'loss_adv': g_loss_adv,
#                         'epoch':epoch,
#                         }, save_path_generator)

#             torch.save({'d_state_dict': d_net.state_dict(),
#                         'optimizer_state_dict': d_optimizer.state_dict(),
#                         'loss_fake': d_fake_loss,
#                         'loss_real': d_real_loss,
#                         'loss': d_loss,
#                         'epoch': epoch
#                         }, save_path_discriminator)
#             if ((epoch+1)%save_photos_interval==0):
#                 compare_and_save(64,path_toSave_photos,test_loader,g_net)
#             print(f"Epoch {epoch+1} DL={epoch_d_loss:.4f} DR={epoch_d_real_loss:.4f} DF={epoch_d_fake_loss:.4f} GL={epoch_g_loss:.4f} GLP={epoch_g_loss_pixel:.4f} GLADV={epoch_g_loss_adv:.4f} Time {time_epoch:.1f}min Total Time: {total_time/60 :.1f}h")
#             # Evaluation mode
#             g_net.eval()
#             with torch.no_grad():
#                 # Removing center from the test sample
#                 sample = apply_center_mask(test_image)
#                 # Forward (generator)
#                 g_sample = g_net(sample)
#                 # Impanting the image generated to the original
#                 test_masked_imgs[:,:,(mask_size//2):img_size-(mask_size//2),(mask_size//2):img_size-(mask_size//2)] = g_sample.data
#                 plt.imshow(TF.to_pil_image(make_grid(test_masked_imgs[:4], scale_each=True, normalize=True).cpu()))
#                 plt.axis('off')
#                 plt.show()
                
    except KeyboardInterrupt:
          print("Interrupted")
            
training(train_loader, last_epoch = 50, restore = True)