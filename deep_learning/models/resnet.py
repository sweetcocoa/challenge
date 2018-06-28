import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.resnet import resnet18
import pretrainedmodels

class CelebNet(nn.Module):
    def __init__(self, num_classes):
        super(CelebNet, self).__init__()
        self.num_classes = num_classes
        self.resnet = resnet18(num_classes=512)
        self.relu = nn.ReLU()
        for i in range(self.num_classes):
            self.add_module(f"seq_class_{i}", self._make_node())


    def _make_node(self):
        seq = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        return seq


    def forward(self, x):
        x = self.relu(self.resnet(x))

        out = []
        for i in range(self.num_classes):
            x_i = self.__getattr__(f"seq_class_{i}")(x)
            out.append(x_i)

        x = torch.cat(out, dim=1)
        # print(x.shape)
        return x


class PretrainedResNet(nn.Module):
    def __init__(self, original_model):
        super(PretrainedResNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def GetPolyNet(nc=1000, modelname='se_resnext50_32x4d'):
    net = pretrainedmodels.__dict__[modelname](num_classes=nc, pretrained='imagenet')
    net.train(True)
    # Finetune Last Layer
    net.last_linear = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
    return net