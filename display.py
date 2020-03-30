import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger
from PIL import Image
import numpy as np 

class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256,512),
            nn.LeakyReLU(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

generator = GeneratorNet()

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    n = Variable(torch.randn(size,100))
    return n

generator.load_state_dict(torch.load('./generator.pt'))
generator.eval()
fake_data = generator(noise(1))
fake_data = (fake_data + 1)*255/2
fake_data = vectors_to_images(fake_data)
fake_data = fake_data.detach().numpy()
fake_data = fake_data[0,0,:,:]
img = Image.fromarray(np.uint8(fake_data),'L')
img.show()

