# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# setting hyperparameters
batchSize = 64
imageSize = 64
imageSize

#Create transformation object (scaling, tensor conversion, normalization) to apply to the input images.
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

# Loading the dataset
dataset = dset.CIFAR10(root= './data', download=True, transform = transform)
# get images by batch
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)

#function to initialize weights of nn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) # weights of conv layer
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) #weights of batch norm layer
        m.bias.data.fill_(0)

# Define the generator
class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        #creating the architecture of the nn
        # we are making a deconvolutional netork
        self.main = nn.Sequential(
            #input, featuremaps, kernels, stride, padding
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512), #normalize the features
            nn.ReLU(True), # ReLu rectification activation

            #another layer of inverse convolution
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256), #normalize the features
            nn.ReLU(True), # ReLu rectification activation

            #another layer of inverse convolution
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128), #normalize the features
            nn.ReLU(True), # ReLu rectification activation

            #another layer of inverse convolution
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64), #normalize the features
            nn.ReLU(True), # ReLu rectification activation

            # output layer
            # 3 channels for the images
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh() #apply TanH to keep output between -1 and 1
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Creating generator
netG = G()
netG.apply(weights_init)

# Define the discriminator
class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            #output layer
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() # this will give a logistic output between 0 and 1
        )

    def forward(self, input):
        output = self.main(input)
        # flatten the result of the convoluation 
        return output.view(-1)

#create the discriminator
netD = D()
netD.apply(weights_init)

# Training the DCGAN
#LOSS criterion
criterion = nn.BCELoss() #(Binary cross entropy loss)
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

if __name__ == '__main__':
    for epoch in range(25): # maybe I can run 50 If I can run this on cloud

        # iterate over each batch
        for i, data in enumerate(dataloader, 0):

            #update weights of discriminator

            netD.zero_grad() # intialize the gradient value

            #train with real images
            real, _ = data
            input = Variable(real) #create Torch variable
            target = Variable(torch.ones(input.size()[0])) # target of only one's since image is real
            output = netD(input)
            errD_real = criterion(output, target)

            #train with fake images
            # creating minibatches of random noise
            # creating 100 feature maps of size 1x1
            noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
            fake = netG(noise)
            target = Variable(torch.zeros(input.size()[0]))
            # detaches the gradient to save memory. We are not using this for 
            # disciminator
            output = netD(fake.detach())
            errD_fake = criterion(output, target)

            errD = errD_fake + errD_real
            errD.backward()
            #updates weights
            optimizerD.step()

            # updating weights of the Generator

            netG.zero_grad()
            target = Variable(torch.ones(input.size()[0]))
            # we are reusing the fake images used to train the discriminator instead of creating new fake images
            ouput = netD(fake)
            # noise_new = Variable(torch.randn(input.size()[0], 100, 1, 1))
            # fake_new = netG(noise_new)
            # output = netD(fake_new)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()

        # print loss and save every 100 steps
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data, errG.data))
        if i % 100 == 0:
            #%s refers to the root in our local worspace
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) # We save the real images of the minibatch.
            fake = netG(noise) # We get our fake generated images.
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) # We also save the fake generated images of the minibatch.
