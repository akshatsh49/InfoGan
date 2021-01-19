from global_variables import *

class generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(74, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)

        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)

    def forward(self,x):
        x = F.relu(self.bn1(self.tconv1(x.view(x.shape[0],74,1,1))))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        img = torch.sigmoid(self.tconv4(x))
        return img

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

        self.mu=nn.Linear(1024,2)
        self.logsigma=nn.Linear(1024,2)
        self.fc=nn.Linear(1024,1)
        self.logits=nn.Linear(1024,10)

    def feature_extractor(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        return x

    def forward(self,x):
      x=self.feature_extractor(x).view(x.shape[0],-1)
      score=nn.Sigmoid()(self.fc(x))
      mu=self.mu(x)
      logsigma=self.logsigma(x)
      logits=nn.Softmax(dim=1)(self.logits(x))
      return score,mu,logsigma,logits

