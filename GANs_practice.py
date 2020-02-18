# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.serialization import load_lua
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from torch.autograd import Variable
matplotlib_is_available = True
try:
  from matplotlib import pyplot as plt
except ImportError:
  print("Will skip plotting; matplotlib is not available.")
  matplotlib_is_available = False



#generate 2D Gaussian data set, class=3,
# cluster center=[[-4,3],[2,7],[-4,-6]], varience=1, num=10000
def generated_gaussian():
    num = 10000
    dim = 2
    sigma = 1
    sample_set = []
    mu_set = [[-4,3],[2,7],[-4,-6]]
    np.random.seed(2020)
    for i, mu in enumerate(mu_set):
        X = np.random.randn(num,dim)*sigma + mu
        y = np.ones(len(X))*i
        y = y.reshape(-1,1)
        sample = np.concatenate((X,y),axis=1)
        sample_set.append(sample)

        def concat(a,b):
            return np.concatenate((a,b),axis=1)
        gaussian = reduce(concat, sample_set)
        np.random.shuffle(gaussian)

    return gaussian


args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(arg.input_dim,128),
            nn.ReLU(True),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        out = self.main(inputs)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(arg.z_dim,128),
            nn.ReLU(True),
            nn.Linear(128,args.input_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        fake_instance = self.main(z)
        return fake_instance


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(args.input_dim,128),
            nn.ReLU(True),
            nn.Linear(128,args.label_dim),
        )

    def forward(self, inputs):
        logits = self.main(inputs)
        out = self.softmax(logits)
        return logits


#training

#Training setting
class Config:
    lr = 0.0002
    beta1 = 0.5
    batch_size = 80
    epochs = 200
    workers = 2
    cuda = True
    z_dim = 10
    input_dim = 2
    lable_dim = 3
    seed = 2020
    wd = 0.0
    decreasing_lr = [60,]
    beta = 1
    log_interval = 10
    drop_rate = 0.1
    outf = './gaussian'

##################
#Update D network#
##################
#train with real
gan_target.fill_(real_label)
targetv = Variable(gan_target)
optimizerD.zero_grad()
output = netD(data.float())
errD_real = criterion(output,targetv)
errD_real.backward()
D_x = output.data.mean()

#train with fake

noise = torch.FloatTensor(data.size(0), args.z_dim).normal_(0,1).cuda()
if args.cuda:
    noise = noise.cuda()
noise = Variable(noise)

fake = netG(noise)
    pdb.set_trace()
targetv = Variable(gan_target.fill_(fake_lable))
output = netD(fake.detach())
errD_fake = criterion(output, targetv)
errD_fake.backward()
D_G_z1 = output.data.mean()
errD = errD_real + errD_fake
optimizerD.step()

##################
#Update G network#
##################
optimizerG.zero_grad()
#Original GAN loss
targetv = Variable(gan_target.fill_(real_label))
output = netD(fake)
errG = criterion(output, targetv)
D_G_z2 = output.data.mean()

#minimize the true distribution
    KL_fake_output = F.log_softmax(model(fake.view(fake.size(0),1,28,28)))
KL_fake_output = F.log_softmax(model(fake),dim=1)
errG_KL = F.kl_div(KL_fake_output,uniform_dist,reduction='batchmean')*args.label_dim
generator_loss = errG + args.beta*errG_KL
generator_loss.backward()
optimizerG.step()


##################
#Update classifier#
##################
#cross entropy loss
optimizer.zero_grad()
output = F.log_softmax(model(data.float()),dim=1)
loss = F.nll_loss(output,target.long())

#KL divergence
noise = torch.FloatTensor(data.size(0), args.z_dim).normal_(0,1).cuda()
if args.cuda:
    noise = noise.cuda()
noise = Variable(noise)
fake = netG(noise)
KL_fake_output = F.log_softmax(model(fake),dim=1)
KL_loss_output = F.kl_div(KL_fake_output,uniform_dist, reduction='batchmean')*args.label_dim
total_loss = loss + args.beta*KL_loss_output
total_loss.backward()
optimizer.step()