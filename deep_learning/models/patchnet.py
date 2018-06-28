import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models.resnet import resnet18
from collections import OrderedDict

class PatchNet(nn.Module):
    def __init__(self, original_model, num_dim=64):
        super(PatchNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, num_dim),
        )

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


class PatchNet2(nn.Module):
    def __init__(self, patchnet, num_dim=256):
        super(PatchNet2, self).__init__()
        self.features = patchnet.features
        self.features[-1] = nn.AvgPool2d(2, stride=1)

        self.cnn = nn.Conv2d(in_channels=512, out_channels=num_dim, kernel_size=1)
        self.cnn.weight = nn.Parameter(patchnet.classifier[0].weight.view(num_dim, 512, 1, 1))
        self.cnn.bias = nn.Parameter(patchnet.classifier[0].bias)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = self.cnn(f)
        return f

class SiamesePatchNetwork(nn.Module):
    def __init__(self, baseNetwork):
        super(SiamesePatchNetwork, self).__init__()
        self.net = baseNetwork

    def forward_once(self, x):
        return self.net(x)

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3


class TripletHingeLoss(nn.Module):
    def __init__(self, margin):
        super(TripletHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletHingeLoss2(nn.Module):
    def __init__(self, margin):
        super(TripletHingeLoss2, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.pairwise_distance(anchor, positive)  # .pow(.5)
        distance_negative = F.pairwise_distance(anchor, negative)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class PatchClassifier(nn.Module):
    def __init__(self, patchnet2):
        super(self.__class__, self).__init__()
        self.patchnet = patchnet2
        for param in self.patchnet.parameters():
            param.requires_grad = False

        self.conv1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.relu1   = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=4)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool2d(1)

        self.fcn1 = nn.Linear(256 + 256 + 128 + 128, 512)
        self.frelu1 = nn.ReLU()
        self.fcn2 = nn.Linear(512, 32)
        self.frelu2 = nn.ReLU()
        self.fcn3 = nn.Linear(32, 1)


    def forward(self, x):
        x = self.patchnet(x)
        x1 = self.relu1(self.conv1_1(x))
        x1 = self.pool(self.conv1_2(x1))
        x2 = self.pool(self.conv2(x))
        x3 = self.pool(self.conv3(x))
        x4 = self.pool(self.conv4(x))

        xs = torch.cat([x1, x2, x3, x4], dim=1).view(x.shape[0], -1)
        xs = self.frelu1(self.fcn1(xs))
        xs = self.frelu2(self.fcn2(xs))
        xs = self.fcn3(xs)
        return xs



class ConvVectorClassifier(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.relu1   = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=4)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool2d(1)

        self.fcn1 = nn.Linear(256 + 256 + 128 + 128, 512)
        self.frelu1 = nn.ReLU()
        self.fcn2 = nn.Linear(512, 32)
        self.frelu2 = nn.ReLU()
        self.fcn3 = nn.Linear(32, 1)


    def forward(self, x):
        x1 = self.relu1(self.conv1_1(x))
        x1 = self.pool(self.conv1_2(x1))
        x2 = self.pool(self.conv2(x))
        x3 = self.pool(self.conv3(x))
        x4 = self.pool(self.conv4(x))

        xs = torch.cat([x1, x2, x3, x4], dim=1).view(x.shape[0], -1)
        xs = self.frelu1(self.fcn1(xs))
        xs = self.frelu2(self.fcn2(xs))
        xs = self.fcn3(xs)
        return xs


class CNNVectorClassifier(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=3)),
                ('conv2', nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=3)),
                ('pool', nn.MaxPool2d(kernel_size=4)),
                ])
            )

        self.fcn = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(512, 128)),
                ('relu1', nn.ReLU(inplace=True)),
                ('linear2', nn.Linear(128, 32)),
                ('relu2', nn.ReLU(inplace=True)),
                ('linear3', nn.Linear(32, 1)),
                ])
        )

    def forward(self, x):
        # print(x.shape)
        x = self.convnet(x)
        x = x.view(x.shape[0], -1)
        x = self.fcn(x)
        return x



class EmbeddedVectorClassifier(nn.Module):
    def __init__(self,
                 embedding_dim=1000,
                 hidden_size=128,
                 bidirectional=False,
                 num_patch=68
                 ):
        super(self.__class__, self).__init__()
        # self.patchnet = patchnet
        # for param in self.patchnet.parameters():
        #     param.requires_grad = False
        # self.patchnet.eval()

        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_size,
                           batch_first=True,
                           bidirectional=bidirectional,
                           num_layers=1)
        self.hidden_size = hidden_size
        self.global_pool = nn.MaxPool1d(hidden_size)
        self.fcn = nn.Sequential(
            OrderedDict([
                ('linear1', nn.Linear(num_patch, 128)),
                ('relu1', nn.ReLU(inplace=True)),
                ('linear2', nn.Linear(128, 32)),
                ('relu2', nn.ReLU(inplace=True)),
                ('linear3', nn.Linear(32, 1))
            ])
        )


    def forward(self, x):
        # x : batch x 68 x 3 x 224 x 224
        # print(x.shape)

        # patches = []
        # for i in range(x.shape[0]):
        #     patch = self.patchnet(x[i])
        #     patches.append(patch)

        # patches = batch x 68 x 1000

        # x = torch.stack(patches)
        # x : batch x 68 x 1000
        # print(x.shape)
        # self.rnn.flatten_parameters()
        x, (h, _) = self.rnn(x)
        # x = batch x 68 x 256
        # print(x.shape, h.shape, _.shape)

        x = self.global_pool(x)
        # x = batch x 68 x 1
        # print(x.shape)

        x = x.view(x.shape[0], -1)
        # x = batch x 68

        x = self.fcn(x)
        #x = batch x 1


        return x

# if __name__ == "__main__":
